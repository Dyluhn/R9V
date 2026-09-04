// SPDX-License-Identifier: Apache-2.0
//! Deterministic launch list record and replay tests, stub execution, and profiling sink hook (Spec 4 §7, §12).

use std::sync::Mutex;

use r9v_registry::{
    dispatch_launch, DeviceExecutor, LaunchEntry, LaunchGeometry, LaunchList, ProfileSink,
    RegistryError, Result, StubDevice, VariantHash,
};

#[derive(Default)]
struct TestProfileSink {
    starts: Mutex<Vec<LaunchEntry>>,
    ends: Mutex<Vec<LaunchEntry>>,
    call_log: Mutex<Vec<String>>,
    fail_start: Mutex<bool>,
    fail_end: Mutex<bool>,
}

impl ProfileSink for TestProfileSink {
    fn record_start(&self, entry: &LaunchEntry) -> Result<()> {
        self.starts.lock().unwrap().push(entry.clone());
        self.call_log
            .lock()
            .unwrap()
            .push("record_start".to_owned());
        if *self.fail_start.lock().unwrap() {
            return Err(RegistryError::LaunchError {
                symbol: entry.entry_symbol.clone(),
                detail: "simulated hipEventRecord start failure".to_owned(),
            });
        }
        Ok(())
    }

    fn record_end(&self, entry: &LaunchEntry) -> Result<()> {
        self.ends.lock().unwrap().push(entry.clone());
        self.call_log.lock().unwrap().push("record_end".to_owned());
        if *self.fail_end.lock().unwrap() {
            return Err(RegistryError::LaunchError {
                symbol: entry.entry_symbol.clone(),
                detail: "simulated hipEventRecord end failure".to_owned(),
            });
        }
        Ok(())
    }
}

#[test]
fn test_deterministic_launch_list_replay() {
    let mut list = LaunchList::new();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);

    let vh1 = VariantHash::new(0x1111222233334444);
    let vh2 = VariantHash::new(0x5555666677778888);
    let vh3 = VariantHash::new(0xaaaabbbbccccdddd);

    list.record(LaunchEntry::new(
        vh1,
        "rmsnorm_kernel",
        LaunchGeometry::new([64, 1, 1], [256, 1, 1], 0),
        0,
        1024,
        2048,
        vec![1, 0, 0, 0],
    ));
    list.record(LaunchEntry::new(
        vh2,
        "matmul_qkv_kernel",
        LaunchGeometry::new([128, 16, 1], [128, 1, 1], 4096),
        8192,
        65536,
        131072,
        vec![2, 0, 0, 0],
    ));
    list.record(LaunchEntry::new(
        vh3,
        "attention_kernel",
        LaunchGeometry::new([32, 1, 1], [64, 1, 1], 16384),
        16384,
        32768,
        65536,
        vec![3, 0, 0, 0],
    ));

    assert_eq!(list.len(), 3);
    assert!(!list.is_empty());

    // First replay against stub device 1
    let stub1 = StubDevice::new();
    list.replay(&stub1, None)
        .expect("first replay should succeed");
    let records1 = stub1.recorded_launches().unwrap();
    assert_eq!(records1.len(), 3);

    // Second replay against stub device 2
    let stub2 = StubDevice::new();
    list.replay(&stub2, None)
        .expect("second replay should succeed");
    let records2 = stub2.recorded_launches().unwrap();
    assert_eq!(records2.len(), 3);

    // Assert that replay is bit-identical
    assert_eq!(records1, records2);

    assert_eq!(records1[0].variant_hash, vh1);
    assert_eq!(records1[0].entry_symbol, "rmsnorm_kernel");
    assert_eq!(records1[0].workspace_used, 0);
    assert_eq!(records1[0].static_bytes, 1024);
    assert_eq!(records1[0].static_flops, 2048);

    assert_eq!(records1[1].variant_hash, vh2);
    assert_eq!(records1[1].entry_symbol, "matmul_qkv_kernel");
    assert_eq!(records1[1].workspace_used, 8192);

    assert_eq!(records1[2].variant_hash, vh3);
    assert_eq!(records1[2].entry_symbol, "attention_kernel");
    assert_eq!(records1[2].workspace_used, 16384);
}

#[test]
fn test_disabled_cost_and_active_profiler_sink() {
    let stub = StubDevice::new();
    let vh = VariantHash::new(0x42);
    let entry = LaunchEntry::new(
        vh,
        "test_kernel",
        LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        128,
        256,
        512,
        vec![0xff],
    );

    // 1. Profiler disabled: None (single forward branch)
    dispatch_launch(&stub, &entry, None).expect("dispatch without profiler must succeed");
    assert_eq!(stub.recorded_launches().unwrap().len(), 1);

    // 2. Profiler enabled: active ProfileSink receives start and end events
    let sink = TestProfileSink::default();
    dispatch_launch(&stub, &entry, Some(&sink)).expect("dispatch with profiler must succeed");
    assert_eq!(stub.recorded_launches().unwrap().len(), 2);

    let starts = sink.starts.lock().unwrap();
    assert_eq!(starts.len(), 1);
    assert_eq!(starts[0].entry_symbol, "test_kernel");

    let ends = sink.ends.lock().unwrap();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].entry_symbol, "test_kernel");

    // Event sequence check: record_start must precede execution, followed by record_end
    let calls = sink.call_log.lock().unwrap();
    assert_eq!(
        calls.as_slice(),
        &["record_start".to_string(), "record_end".to_string()]
    );
}

struct FailingDevice;

impl DeviceExecutor for FailingDevice {
    fn launch_kernel(&self, entry: &LaunchEntry) -> Result<()> {
        Err(RegistryError::LaunchError {
            symbol: entry.entry_symbol.clone(),
            detail: "simulated kernel execution failure".to_owned(),
        })
    }
}

#[test]
fn test_profile_sink_error_propagation_and_launch_error_handling() {
    let stub = StubDevice::new();
    let entry = LaunchEntry::new(
        VariantHash::new(0x99),
        "err_kernel",
        LaunchGeometry::new([1, 1, 1], [64, 1, 1], 0),
        0,
        0,
        0,
        vec![],
    );

    // 1. Profiler start failure aborts before kernel launch
    let sink_start_err = TestProfileSink::default();
    *sink_start_err.fail_start.lock().unwrap() = true;
    let res = dispatch_launch(&stub, &entry, Some(&sink_start_err));
    assert!(res.is_err());
    match res.unwrap_err() {
        RegistryError::LaunchError { detail, .. } => {
            assert!(detail.contains("simulated hipEventRecord start failure"));
        }
        other => panic!("expected LaunchError, got {other:?}"),
    }
    // Launch was never called
    assert_eq!(stub.recorded_launches().unwrap().len(), 0);
    // record_end was never called
    assert_eq!(sink_start_err.ends.lock().unwrap().len(), 0);

    // 2. Kernel launch failure invokes record_end to balance stream event pairs and returns launch error
    let failing_dev = FailingDevice;
    let sink_launch_err = TestProfileSink::default();
    let res2 = dispatch_launch(&failing_dev, &entry, Some(&sink_launch_err));
    assert!(res2.is_err());
    match res2.unwrap_err() {
        RegistryError::LaunchError { detail, .. } => {
            assert!(detail.contains("simulated kernel execution failure"));
        }
        other => panic!("expected LaunchError from launch, got {other:?}"),
    }
    // Both start and end were recorded to keep the profiler stream event pair balanced
    assert_eq!(sink_launch_err.starts.lock().unwrap().len(), 1);
    assert_eq!(sink_launch_err.ends.lock().unwrap().len(), 1);

    // 3. Profiler end failure propagates error when launch succeeds
    let sink_end_err = TestProfileSink::default();
    *sink_end_err.fail_end.lock().unwrap() = true;
    let res3 = dispatch_launch(&stub, &entry, Some(&sink_end_err));
    assert!(res3.is_err());
    match res3.unwrap_err() {
        RegistryError::LaunchError { detail, .. } => {
            assert!(detail.contains("simulated hipEventRecord end failure"));
        }
        other => panic!("expected LaunchError from end, got {other:?}"),
    }
    // Launch succeeded on stub
    assert_eq!(stub.recorded_launches().unwrap().len(), 1);
}

#[test]
fn test_launch_list_serde_roundtrip() {
    let mut list = LaunchList::new();
    list.record(LaunchEntry::new(
        VariantHash::new(0xabcdef),
        "serde_kernel",
        LaunchGeometry::new([2, 1, 1], [64, 1, 1], 0),
        64,
        128,
        256,
        vec![10, 20],
    ));

    let json = serde_json::to_string(&list).expect("serialize launch list");
    let deserialized: LaunchList = serde_json::from_str(&json).expect("deserialize launch list");
    assert_eq!(list, deserialized);

    let stub = StubDevice::new();
    deserialized
        .replay(&stub, None)
        .expect("replay deserialized");
    assert_eq!(stub.recorded_launches().unwrap().len(), 1);
}
