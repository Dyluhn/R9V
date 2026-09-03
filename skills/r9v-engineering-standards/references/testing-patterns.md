# Testing patterns

## The op harness (card A1.10)

Every op-level implementation, on any tier, is tested through the shared harness. It owns fixture generation, the tolerance table, and the three invariance checks.

```rust
use r9v_t0::harness::{self, Tier, Tolerance};

#[test]
fn matmul_i4k_pertoken_t1_matches_t0() {
    let op = fixtures::matmul(Scheme::I4K, Act::PerTokenI8, /*M*/ 8, /*N*/ 4096, /*K*/ 4096);
    harness::golden(&op, &t1::matmul, Tier::T0, Tolerance::for_op(op.id()));   // 32 seeded inputs + edge shapes
    harness::batch_invariant(&op, &t1::matmul);                                // alone / padded / embedded
    harness::deterministic(&op, &t1::matmul);                                  // twice, bit-equal
}

#[test]
fn matmul_t1_shape_fuzz() {
    harness::shape_fuzz(OpId::Matmul, &t1::matmul, /*cases*/ 64, /*seed*/ 0xA3_04);
}
```

Tolerances come from `Tolerance::for_op`, which reads the spec 1 §6.1 table as data. Never pass a literal.

## Property tests (laws)

Use `proptest` with a fixed seed configured in `CONVENTIONS.md`.

```rust
proptest! {
    #![proptest_config(fixtures::proptest_cfg())]
    #[test]
    fn l1_permute_roundtrip(t in fixtures::any_l1_tensor()) {
        let packed = layout::to_l1(&t);
        let back   = layout::from_l1(&packed, t.shape(), t.dtype());
        prop_assert_eq!(back.bytes(), t.bytes());
    }

    #[test]
    fn repack_never_requantizes(src in fixtures::any_gguf_tensor(GgmlType::Q4K)) {
        let repacked = repack::from_gguf(&src)?;
        prop_assert_eq!(dequant::reference(&src), dequant::reference_l1(&repacked));   // bit-equal f32
    }

    #[test]
    fn commit_rollback_equivalence(k in 1u32..=15, accepted in 0u32..=15) {
        prop_assume!(accepted <= k + 1);
        let a = fixtures::seq_after(|m, s| { m.reserve(s, k + 1)?; write_all(m, s, k + 1); m.commit(s, accepted) });
        let b = fixtures::seq_after(|m, s| { m.reserve(s, accepted)?; write_all(m, s, accepted); m.commit(s, accepted) });
        prop_assert_eq!(a.state_bytes(), b.state_bytes());
    }
}
```

## Golden files

For structured outputs (partitioner graphs, `LayerSpec` lists, load reports, config skeletons):

- Stored under `tests/golden/<crate>/<name>.<json|toml>`.
- Compared with a normalized serialization (sorted keys, no timestamps).
- Regenerated only with `R9V_UPDATE_GOLDEN=1 cargo test -p <crate>`, and the implementing agent audits the regenerated diff. A golden update with no explanation in the PR fails acceptance.

```rust
#[test]
fn partition_pp2_reference_dense() {
    let out = partition(&fixtures::graph_dense_4b(), &fixtures::plan_pp2(), &fixtures::topo_two_r9700());
    golden::assert_matches("part/pp2_reference_dense.json", &out.summary());
}
```

## Failure paths

Every error variant with data has a test that asserts the data:

```rust
#[test]
fn budget_refusal_reports_shortfall_and_suggestion() {
    let err = load::budget(&fixtures::model_30b_summary(), &fixtures::device_with_vram(24 << 30), &cfg).unwrap_err();
    match err {
        LoaderError::Budget { shortfall, largest, suggestion, .. } => {
            assert_eq!(shortfall, expected_shortfall);
            assert_eq!(largest.len(), 5);
            assert!(matches!(suggestion, Suggestion::LowerMaxCtx { .. }));
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

#[test]
fn missing_tensors_are_all_reported() {
    let err = bind(&fixtures::gguf_missing(&["blk.3.attn_q.weight", "output_norm.weight"])).unwrap_err();
    let names: Vec<_> = err.tensor_problems().map(|p| p.name()).collect();
    assert_eq!(names, ["blk.3.attn_q.weight", "output_norm.weight"]);   // both, in order
}
```

Malformed-input tests: truncated file, wrong alignment, bad checksum, unknown `ggml_type`, unknown `general.architecture`. Each asserts the error names the thing.

## Fakes

`FakeDevice` implements the real `Device` trait in host memory and records placements. Tests assert on bytes and regions, not on call order.

```rust
#[test]
fn native_load_is_zero_copy() {
    let mut dev = FakeDevice::new(32 << 30);
    let report = Loader::new(&mut dev, cfg).load(fixtures::native_file())?;
    assert!(report.tensors.iter().all(|t| t.path == LoadPath::ZeroCopy));
    assert_eq!(dev.bytes_at(report.tensor("blk.0.attn_q.weight").region), fixtures::native_file().tensor_bytes("blk.0.attn_q.weight"));
}
```

## Determinism tests

```rust
#[test]
fn schedule_is_reproducible() {
    let run = |seed| { let mut s = Scheduler::new(fixtures::cost_table(), cfg.clone()); s.run(fixtures::requests(seed), 1000).schedule_log() };
    assert_eq!(run(7), run(7));
}
```

## GPU tests

Under `tests/gpu/`, gated on device presence, never silently skipped:

```rust
#[test]
fn decode_64_tokens_t1_matches_t0_within_l1() {
    let Some(dev) = gpu::device_or_skip("gfx1201") else { return gpu::skipped_reported("no gfx1201") };
    let logits_gpu = engine::decode(&dev, fixtures::small_dense_gguf(), fixtures::prompt_128(), 64, Tier::T1)?;
    let logits_cpu = eval::logits(fixtures::small_dense_gguf(), fixtures::prompt_128(), 64)?;
    harness::assert_within(Tolerance::logits(), &logits_gpu, &logits_cpu);
}
```

`gpu::skipped_reported` writes a line to the test output that the runner workflow turns into a failure if the runner should have had the device.

## What not to do

- `assert_eq!(f(x), f(x))` — compares the implementation to itself.
- `assert!((a - b).abs() < 1e-3)` with a literal tolerance.
- A test that reads `/tmp`, the clock, or `HOME`.
- `#[ignore]` on a test the card names.
- A single test that exercises ten behaviors and fails with "assertion failed".
- Mocks that assert `alloc` was called before `copy_h2d`.
