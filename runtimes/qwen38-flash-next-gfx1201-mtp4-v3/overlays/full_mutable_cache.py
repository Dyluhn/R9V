"""Opt-in clone-only integration for the physically qualified cooperative cache."""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch

SO_SHA='ebd7bd39c59f9a54aa6945b636906b1ecc16ed63f62fe39639078c6dc225b4f7'
PINS_SHA='537d7d5b1368fe31057fa07f1114ed1ceec89cff8172192ad6e8945ed7c05cff'
# Warmstart slots [0, pinned) are never evicted by the planner (compiled into the binary).
PINNED_SLOTS={0:0,1:400}
_STATES={}
_EXT=None

def enabled():
    value=os.environ.get('R9V_FULL_MUTABLE_CACHE','0')
    if value not in ('0','1'):raise RuntimeError('Invalid full mutable enable flag')
    return value=='1'

@functools.cache
def pins(path=None):
    # Permanent pins per rank and layer, from tools/pin_sim/make_pins.py (see the file's provenance).
    path=Path(path or os.environ['R9V_FULL_MUTABLE_PINS'])
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS_SHA:raise RuntimeError(f'Unreviewed full mutable pin list: {path}')
    data=json.loads(raw)
    if data['version']!=1 or data['pinned_slots']!={str(r):n for r,n in PINNED_SLOTS.items()}:raise RuntimeError('Pin list does not match the binary')
    result={}
    for rank,count in PINNED_SLOTS.items():
        layers=data['pinned_experts_by_layer'][str(rank)]
        if len(layers)!=48 or any(len(ids)!=count or len(set(ids))!=count or not all(0<=e<512 for e in ids) for ids in layers):
            raise RuntimeError(f'Malformed pin list for rank {rank}')
        result[rank]=tuple(tuple(ids) for ids in layers)
    return result

def hot_ids_for_rank(ids,rank,layer,pin_table=None):
    # Warmstart order: pinned experts first (slots the planner never evicts), then placement priority.
    if not enabled():return ids
    if rank not in (0,1) or len(ids)!=(62 if rank==0 else 428):raise RuntimeError('Unexpected baseline placement')
    pinned=list((pin_table or pins())[rank][layer])
    return (pinned+[e for e in ids if e not in set(pinned)])[:62 if rank==0 else 427]

def host_ids(rank,layer,pin_table=None):
    # Experts the host copy holds: every expert that can ever be non-resident.
    pinned=set((pin_table or pins())[rank][layer])
    return [e for e in range(512) if e not in pinned]

def check_maps(hot_map,cold_map,hot_ids,rank):
    # Pinned experts sit in their warmstart slot and have no host row; every other expert
    # has one, in ascending expert order.
    pinned=hot_ids[:PINNED_SLOTS[rank]]
    if any(hot_map[e]!=slot or cold_map[e]!=-1 for slot,e in enumerate(pinned)):raise RuntimeError('Pinned expert is not resident or has a host row')
    rows=[cold_map[e] for e in range(512) if e not in set(pinned)]
    if rows!=list(range(512-len(pinned))):raise RuntimeError('Host copy row index is not dense and ordered')

def cache_slots_for_rank(rank):
    if rank not in (0,1):raise RuntimeError('Expected TP2')
    required={'QWEN38_TIERED_EXPERT_CACHE_SLOTS':'160','QWEN38_TIERED_EXPERT_CACHE_RANKS':'0','QWEN38_TIERED_EXPERT_CACHE_POLICY':'lru','QWEN38_TIERED_EXPERT_CACHE_ASYNC':'0','R9V_CACHE_FILL_BATCH':'1','QWEN38_TIERED_IQ_MOE_VARIANT':'reuse3v2'}
    if any(os.environ.get(k)!=v for k,v in required.items()):raise RuntimeError('Full mutable baseline configuration mismatch')
    return 155 if rank==0 else 0

def alias_on_device(view,owner,device):
    device=torch.device(device)
    if view.device==device:return view
    if not owner.is_pinned() or owner.dtype!=torch.uint8 or not owner.is_contiguous():raise RuntimeError('Invalid pinned owner')
    make=getattr(torch._C,'_construct_storage_from_data_pointer',None)
    if make is None:raise RuntimeError('Mapped-pointer constructor unavailable')
    storage=make(view.data_ptr(),device,owner.numel()*owner.element_size())
    result=torch.empty(0,dtype=owner.dtype,device=device).set_(storage,0,owner.shape,owner.stride())
    if result.data_ptr()!=view.data_ptr():raise RuntimeError('UVA alias copied unexpectedly')
    return result

def extension():
    global _EXT
    if _EXT is None:
        path=Path(os.environ['R9V_FULL_MUTABLE_SO'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SO_SHA:raise RuntimeError('Unreviewed full mutable binary')
        spec=importlib.util.spec_from_file_location('qwen38_full_mutable_device',path)
        _EXT=importlib.util.module_from_spec(spec);spec.loader.exec_module(_EXT)
    return _EXT

def register(module,rank,hot_ids):
    if not enabled():raise RuntimeError('Registration requires opt-in')
    hotmap=module._gguf_global_to_hot
    key=(hotmap.device.index,hotmap.data_ptr())
    if key in _STATES:raise RuntimeError('Duplicate full mutable registration')
    if len(hot_ids)!=(62 if rank==0 else 427):raise RuntimeError('Wrong warmstart length')
    cold13,cold2=module.w13_qweight,module.w2_qweight
    if cold13.shape[0]!=512-PINNED_SLOTS[rank] or cold2.shape[0]!=512-PINNED_SLOTS[rank]:raise RuntimeError('Host copy size does not match the pin count')
    if cold13.device!=hotmap.device or cold2.device!=hotmap.device:raise RuntimeError('UVA device metadata mismatch')
    arena=torch.zeros(32768,dtype=torch.uint8,device=hotmap.device)
    module.register_buffer('_r9v_mutable_arena',arena,persistent=False)
    ids=torch.tensor(hot_ids,dtype=torch.int32,device=hotmap.device)
    ext=extension()
    ext.full_mutable_device_init(arena,ids,hotmap,module._gguf_global_to_cold,getattr(module,'_gguf_global_to_cache',None),rank,217 if rank==0 else 427)
    # Startup-only completion before transient IDs are released. No runtime sync.
    torch.cuda.synchronize(hotmap.device)
    check_maps(hotmap.tolist(),module._gguf_global_to_cold.tolist(),hot_ids,rank)
    _STATES[key]=(module,rank)

def prepare(hotmap,ids,num_tokens):
    key=(hotmap.device.index,hotmap.data_ptr())
    if key not in _STATES:raise RuntimeError('Unregistered mutable expert layer')
    # Other decode shapes and prefill read existing residency without mutation.
    # The host copy holds every expert that is not pinned resident.
    if num_tokens not in (1,5):return
    module,rank=_STATES[key]
    if ids.shape!=(num_tokens,10):raise RuntimeError('Unexpected routing shape')
    extension().full_mutable_device_step(module._r9v_mutable_arena,ids,
        module.w13_qweight,module.w2_qweight,module._gguf_hot_w13,module._gguf_hot_w2,
        hotmap,module._gguf_global_to_cold,getattr(module,'_gguf_cache_w13',None),
        getattr(module,'_gguf_cache_w2',None),getattr(module,'_gguf_global_to_cache',None),
        rank,217 if rank==0 else 427,64)
