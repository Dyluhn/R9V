# Rust patterns

Concrete shapes for the rules in SKILL.md §2–§3. Match these; do not reinvent them per crate.

## Error types

One `thiserror` enum per crate; variants carry the data a person needs to act; `r9v-common::R9vError` wraps them at the top.

```rust
#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("device {device}: required {required} B, available {available} B, shortfall {shortfall} B; largest: {largest:?}; suggestion: {suggestion}")]
    Budget { device: Rank, required: u64, available: u64, shortfall: u64, largest: Vec<(String, u64)>, suggestion: Suggestion },

    #[error("{} tensor(s) missing or mis-shaped: {details:?}", details.len())]
    Tensors { details: Vec<TensorProblem> },        // every failure, collected

    #[error("checksum mismatch for tensor {name}: expected {expected:016x}, got {actual:016x}")]
    Checksum { name: String, expected: u64, actual: u64 },

    #[error(transparent)]
    Format(#[from] r9v_format::FormatError),
}
```

Collect-all validation:

```rust
let mut problems = Vec::new();
for w in spec.weights() {
    match file.tensor(&w.name) {
        None => problems.push(TensorProblem::Missing { name: w.name.clone() }),
        Some(t) if t.shape != w.shape => problems.push(TensorProblem::Shape { name: w.name.clone(), expected: w.shape, actual: t.shape }),
        Some(_) => {}
    }
}
if !problems.is_empty() { return Err(LoaderError::Tensors { details: problems }); }
```

## Newtypes and closed sets

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(u32);                  // never a bare u32 in a signature

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]     // stable names, never discriminants
pub enum CacheDtype { E4m3, I8, F16 }

// Exhaustive on purpose: adding a variant must break every site that matters.
fn cache_bytes(d: CacheDtype, elems: usize) -> usize {
    match d {
        CacheDtype::E4m3 | CacheDtype::I8 => elems,
        CacheDtype::F16 => elems * 2,
    }
}
```

## Opaque handles

```rust
pub struct StateHandle { layer: LayerIdx, kind: StateKind, group: GroupIdx }   // no pub fields
impl StateHandle {
    pub fn kind(&self) -> StateKind { self.kind }
    pub fn group(&self) -> GroupIdx { self.group }
    // only r9v-state constructs one
    pub(crate) fn new(layer: LayerIdx, kind: StateKind, group: GroupIdx) -> Self { Self { layer, kind, group } }
}
```

## Ordered containers

```rust
// Lookup: fine.
let by_name: HashMap<&str, &TensorInfo> = ...;
let t = by_name.get("blk.0.attn_q.weight");

// Iteration into output: never HashMap. Sort first or use BTreeMap.
let mut names: Vec<&str> = by_name.keys().copied().collect();
names.sort_unstable();
for n in names { report.push(n); }
```

## Arenas and no-alloc step path

```rust
pub struct Arena { base: DevicePtr, len: usize, cursor: usize }
impl Arena {
    /// Reserve `bytes` at 256-byte alignment. Fails with the shortfall; never grows.
    pub fn reserve(&mut self, bytes: usize) -> Result<Region, ArenaError> { ... }
}

// Step path: buffers are fields, filled in place.
pub struct StepBuffers { token_ids: Vec<u32>, slot_map: Vec<u32>, /* capacity = max bucket */ }
impl StepBuffers {
    pub fn fill(&mut self, seqs: &[Sequence]) { self.token_ids.clear(); /* push without realloc */ }
}
```

## Fakes over mocks

```rust
pub trait Device {
    fn alloc(&mut self, bytes: usize) -> Result<DevicePtr, HipError>;
    fn copy_h2d(&mut self, dst: DevicePtr, src: &[u8], stream: Stream) -> Result<(), HipError>;
    fn launch(&mut self, k: &Kernel, args: &[u8], geom: Geometry, stream: Stream) -> Result<(), HipError>;
    // ...
}

/// Host-memory implementation used by cpu-only tests; records placement so tests can assert bytes.
pub struct FakeDevice { mem: Vec<u8>, launches: Vec<LaunchRecord> }
impl Device for FakeDevice { /* real semantics, in RAM */ }
```

Tests assert on outcomes (`fake.bytes_at(region) == expected`), not on which methods were called in what order.

## `unsafe`

```rust
/// Copy `src` into device memory at `dst`. Safe wrapper; the only unsafe is the FFI call.
pub fn memcpy_h2d(dst: DevicePtr, src: &[u8], stream: Stream) -> Result<(), HipError> {
    // SAFETY: `dst` was returned by `hipMalloc` for at least `src.len()` bytes (Arena guarantees
    // this); `src` outlives the call because we synchronize on `stream` before returning to the
    // caller's borrow scope; `hipMemcpyAsync` does not retain the pointer after completion.
    let rc = unsafe { (self.sym.memcpy_async)(dst.0, src.as_ptr().cast(), src.len(), H2D, stream.0) };
    HipError::check(rc)
}
```

## Builders

```rust
let cfg = BenchConfig::builder()
    .suite(Suite::Decode)
    .repeats(5)
    .warmup(2)
    .compare(Compare::LlamaCpp { dir })
    .build()?;                       // validation happens once, here
```

## Tracing

```rust
tracing::info!(req_id = %req.id, step_id = step.id, s = step.s, t_dec = step.t_dec, t_pre = step.t_pre, "step admitted");
// never: token ids, prompt text, completion text at info or above
```

## Pure functions

```rust
/// Spec 5 §4.1. Pure: (graph, plan, topology) -> per-rank graphs. No I/O, no logging.
pub fn partition(graph: &Graph, plan: &Plan, topo: &Topology) -> Result<Vec<RankGraph>, PartError> { ... }
```

Callers log the summary; the function returns it.

## Doc comment shape

```rust
/// Spec 3 §3.6. Commit `accepted` tokens of the current tail.
///
/// Advances `ctx_len` by `accepted` and clears the tail. Positions beyond the new `ctx_len`
/// remain allocated and are overwritten by the next `reserve`. Blocks that became full are
/// hashed into the prefix cache.
///
/// Errors: `StateError::AcceptExceedsTail` if `accepted > tail_len`.
pub fn commit(&mut self, seq: SeqId, accepted: u32) -> Result<(), StateError>
```
