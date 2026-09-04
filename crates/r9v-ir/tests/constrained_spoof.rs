// SPDX-License-Identifier: Apache-2.0
//! End-to-end constrained-device planning flow (Spec 1 App. A, Spec 5 §2,
//! Spec 14 §3): physical discovery facts stay truthful while planning,
//! provenance and the pre-queue launch contract run off the reduced view.
//!
//! The effective planning view is `EffectiveDeviceView`, never an ordinary
//! `DeviceDescriptor`: it carries no measured performance and no P2P links,
//! and every planning number travels beside `Provenance::Spoof`.

use r9v_ir::{
    spoof_catalog, spoof_lookup, ArchDescriptor, ConstrainedDevice, CuMask, DeviceDescriptor,
    DeviceFacts, DeviceIdentity, EffectiveDeviceView, GraphCapture, IrError, Measured,
    PreQueueLaunchContract, Provenance, SpoofProfileId,
};

const GIB: u64 = 1024 * 1024 * 1024;

/// Synthetic bench card: gfx1201 ISA with more resources than either spoof
/// profile, standing in for discovered physical facts.
fn bench_card() -> DeviceDescriptor {
    DeviceDescriptor {
        arch: ArchDescriptor::gfx1201(),
        facts: DeviceFacts {
            identity: DeviceIdentity::Gpu {
                uuid: Some([0x77u8; 16]),
                pci_bdf: "0000:43:00.0".to_owned(),
            },
            cu_count: 96,
            vram_bytes: 32 * GIB,
            l2_bytes: None,
            l3_bytes: None,
            nominal_mem_bw_gbps: None,
            clock_mhz: None,
            graph_capture: GraphCapture::Supported,
        },
        measured: Measured::empty(),
        p2p: Vec::new(),
    }
}

/// Synthetic exact-size R9700-class card: gfx1201, 64 CUs, 32 GiB. It matches
/// the RX 9070 XT (SPOOF) profile CU-for-CU and covers the RX 9070 (SPOOF)
/// profile with headroom, so both profiles constrain cleanly off one device.
fn r9700_card() -> DeviceDescriptor {
    let mut card = bench_card();
    card.facts.cu_count = 64;
    card.facts.vram_bytes = 32 * GIB;
    card
}

#[test]
fn spoof_planning_flow_preserves_truth_and_constrains_effective_view() {
    let physical = bench_card();
    for id in SpoofProfileId::all() {
        let view = ConstrainedDevice::constrain(&physical, id)
            .unwrap_or_else(|e| panic!("bench card covers {id}: {e}"));
        let row = spoof_lookup(id);

        // Physical side stays truthful and separately accessible.
        assert_eq!(view.physical().facts.cu_count, 96);
        assert_eq!(view.physical().facts.vram_bytes, 32 * GIB);
        assert_eq!(view.physical().facts.identity, physical.facts.identity);

        // Effective side is the reduced view: bounds match the catalog row,
        // ISA and identity carry over, provenance is always spoof.
        let effective: &EffectiveDeviceView = view.effective();
        assert_eq!(effective.cu_count(), row.cu_count);
        assert_eq!(effective.vram_bytes(), row.vram_bytes);
        assert_eq!(effective.arch(), &physical.arch);
        assert_eq!(effective.identity(), &physical.facts.identity);
        assert_eq!(effective.profile(), id);

        // Provenance qualifies the target and preserves identity; the view
        // and the pair agree.
        let provenance = view.provenance();
        assert!(provenance.is_spoof());
        let rendered = provenance.to_string();
        assert!(rendered.contains(" (SPOOF) on "), "{rendered}");
        assert!(rendered.contains("0000:43:00.0"), "{rendered}");
        assert_eq!(provenance.physical_identity(), &physical.facts.identity);
        assert_eq!(effective.provenance(), provenance);

        // Launch contract: deterministic mask the launcher applies.
        let contract = PreQueueLaunchContract::for_constrained(&view);
        contract
            .validate_against(&view)
            .expect("contract validates against its own device");
        let (name, value) = contract
            .env_assignment()
            .expect("reduced 96-CU card always needs a mask");
        assert_eq!(name, "ROC_GLOBAL_CU_MASK");
        assert_eq!(CuMask::parse(&value).unwrap(), contract.mask());
        assert_eq!(contract.mask().cu_count(), row.cu_count);
        assert!(contract.validate_env_value(Some(&value)).is_ok());
    }
    assert_eq!(spoof_catalog().len(), 2);
}

#[test]
fn r9700_exact_card_constrains_xt_profile_with_no_mask() {
    // Exact-CU hardware: physical CUs equal the XT bound, so the queue needs
    // no ROC_GLOBAL_CU_MASK assignment at all.
    let physical = r9700_card();
    assert_eq!(physical.arch.name, "gfx1201");
    let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070Xt)
        .expect("64-CU R9700 card exactly covers the XT profile");
    assert_eq!(view.cu_reduction(), 0);
    assert_eq!(view.effective().cu_count(), 64);
    assert_eq!(view.effective().vram_bytes(), 16 * GIB);
    assert_eq!(view.vram_reduction_bytes(), 16 * GIB);
    assert_eq!(view.effective().identity(), &physical.facts.identity);

    let contract = PreQueueLaunchContract::for_constrained(&view);
    assert_eq!(contract.physical_cus(), 64);
    assert_eq!(contract.effective_cus(), 64);
    assert!(!contract.requires_mask());
    assert_eq!(contract.env_assignment(), None);
    // Unset variable validates; any supplied value is a typed refusal.
    assert!(contract.validate_env_value(None).is_ok());
    assert!(matches!(
        contract.validate_env_value(Some("0xffffffffffffffff")),
        Err(IrError::UnexpectedCuMask { cus: 64, .. })
    ));
}

#[test]
fn r9700_exact_card_constrains_base_profile_with_low_n_mask() {
    // Same card against the 56-CU profile: reduced, deterministic lowest-N
    // mask, validated before queue creation.
    let physical = r9700_card();
    let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070)
        .expect("64-CU R9700 card covers the 56-CU profile");
    assert_eq!(view.cu_reduction(), 8);
    assert_eq!(view.effective().cu_count(), 56);

    let contract = PreQueueLaunchContract::for_constrained(&view);
    assert!(contract.requires_mask());
    let (name, value) = contract
        .env_assignment()
        .expect("reduced target needs a mask");
    assert_eq!(name, "ROC_GLOBAL_CU_MASK");
    assert_eq!(value, "0xffffffffffffff");
    assert!(contract.validate_env_value(Some(&value)).is_ok());
    assert!(matches!(
        contract.validate_env_value(None),
        Err(IrError::MissingCuMask {
            effective_cus: 56,
            ..
        })
    ));
    // A full 64-CU mask is well-formed but wrong for this plan.
    assert!(matches!(
        contract.validate_env_value(Some("0xffffffffffffffff")),
        Err(IrError::CuMaskMismatch {
            mask_cus: 64,
            effective_cus: 56,
            ..
        })
    ));
}

#[test]
fn launch_contract_cannot_cross_physical_cu_shapes() {
    let exact = ConstrainedDevice::constrain(&r9700_card(), SpoofProfileId::Rx9070Xt)
        .expect("64-CU card covers XT profile");
    let larger = ConstrainedDevice::constrain(&bench_card(), SpoofProfileId::Rx9070Xt)
        .expect("96-CU card covers XT profile");
    let contract = PreQueueLaunchContract::for_constrained(&exact);
    let error = contract
        .validate_against(&larger)
        .expect_err("exact-CU contract cannot be reused where a mask is required");
    assert!(matches!(
        error,
        IrError::SpoofLaunchContractMismatch {
            field: "physical_cus",
            ..
        }
    ));
}

#[test]
fn spoof_launch_validates_live_process_env_without_mutating_it() {
    // The only test that touches the process environment: it saves, sets,
    // and restores ROC_GLOBAL_CU_MASK, so parallel tests never observe it.
    struct Restore {
        previous: Option<String>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("ROC_GLOBAL_CU_MASK", value),
                None => std::env::remove_var("ROC_GLOBAL_CU_MASK"),
            }
        }
    }
    let _restore = Restore {
        previous: std::env::var("ROC_GLOBAL_CU_MASK").ok(),
    };

    let reduced = PreQueueLaunchContract::for_constrained(
        &ConstrainedDevice::constrain(&r9700_card(), SpoofProfileId::Rx9070)
            .expect("covers the base profile"),
    );
    std::env::remove_var("ROC_GLOBAL_CU_MASK");
    assert!(matches!(
        reduced.validate_process_env(),
        Err(IrError::MissingCuMask { .. })
    ));
    std::env::set_var("ROC_GLOBAL_CU_MASK", "0xffffffffffffff");
    assert!(reduced.validate_process_env().is_ok());
    std::env::set_var("ROC_GLOBAL_CU_MASK", "0xffffffffffffffff");
    assert!(matches!(
        reduced.validate_process_env(),
        Err(IrError::CuMaskMismatch { .. })
    ));

    let exact = PreQueueLaunchContract::for_constrained(
        &ConstrainedDevice::constrain(&r9700_card(), SpoofProfileId::Rx9070Xt)
            .expect("exactly covers the XT profile"),
    );
    std::env::remove_var("ROC_GLOBAL_CU_MASK");
    assert!(exact.validate_process_env().is_ok());
    std::env::set_var("ROC_GLOBAL_CU_MASK", "0xffffffffffffffff");
    assert!(matches!(
        exact.validate_process_env(),
        Err(IrError::UnexpectedCuMask { .. })
    ));
}

#[test]
fn spoof_flow_refuses_undersized_physical_device() {
    let mut small = bench_card();
    small.facts.cu_count = 48;
    small.facts.vram_bytes = 12 * GIB;
    // Both profiles refuse a card below either bound, collecting every
    // shortfall rather than stopping at the first.
    for id in SpoofProfileId::all() {
        let err = ConstrainedDevice::constrain(&small, id)
            .expect_err("48-CU / 12 GiB card cannot cover either profile");
        match err {
            IrError::Multiple { problems } => assert_eq!(problems.len(), 2),
            other => panic!("expected collect-all refusal, got: {other:?}"),
        }
    }
    // Physical provenance carries no spoof target.
    let physical = Provenance::Physical {
        identity: small.facts.identity.clone(),
    };
    assert_eq!(physical.target_label(), None);
}

#[test]
fn spoof_flow_reports_exact_shortfalls_per_profile() {
    // One CU short of XT still covers the base profile's CU bound but not
    // its VRAM: each profile reports exactly its own shortfalls.
    let mut card = r9700_card();
    card.facts.cu_count = 63;
    card.facts.vram_bytes = 12 * GIB;
    match ConstrainedDevice::constrain(&card, SpoofProfileId::Rx9070Xt).unwrap_err() {
        IrError::Multiple { problems } => {
            assert_eq!(problems.len(), 2);
            assert!(matches!(
                &problems[0],
                IrError::SpoofInsufficientCus {
                    required_cus: 64,
                    physical_cus: 63,
                    shortfall_cus: 1,
                    ..
                }
            ));
        }
        other => panic!("expected collect-all refusal, got: {other:?}"),
    }
    match ConstrainedDevice::constrain(&card, SpoofProfileId::Rx9070).unwrap_err() {
        IrError::SpoofInsufficientVram {
            required_bytes,
            physical_bytes,
            shortfall_bytes,
            ..
        } => {
            assert_eq!(required_bytes, 16 * GIB);
            assert_eq!(physical_bytes, 12 * GIB);
            assert_eq!(shortfall_bytes, 4 * GIB);
        }
        other => panic!("expected single VRAM refusal, got: {other:?}"),
    }
}

#[test]
fn spoof_provenance_cannot_be_lost_or_mistaken_for_physical() {
    // Every planning number out of the view arrives beside spoof provenance
    // carrying the qualified MODEL (SPOOF) label and the physical identity;
    // the truthful descriptor stays separately reachable with full numbers.
    let physical = r9700_card();
    for id in SpoofProfileId::all() {
        let view = ConstrainedDevice::constrain(&physical, id)
            .unwrap_or_else(|e| panic!("R9700 card covers {id}: {e}"));
        let effective = view.effective();
        assert!(effective.provenance().is_spoof());
        assert!(!effective.provenance().is_physical());
        assert_eq!(effective.provenance(), view.provenance());
        assert_eq!(
            effective.provenance().physical_identity(),
            &physical.facts.identity
        );
        let label = effective
            .provenance()
            .target_label()
            .expect("spoof provenance always names its target");
        assert!(label.ends_with(" (SPOOF)"), "{label}");
        // Truth is not reduced: the physical descriptor keeps full numbers
        // under the same identity.
        assert_eq!(view.physical().facts.cu_count, 64);
        assert_eq!(view.physical().facts.vram_bytes, 32 * GIB);
        assert_eq!(view.physical().facts.identity, physical.facts.identity);
    }
}

#[test]
fn spoof_results_refuse_official_qualification_with_typed_errors() {
    // A disclaimer string alone is insufficient: official use must fail with
    // a typed refusal the caller cannot ignore.
    let physical = r9700_card();
    for id in SpoofProfileId::all() {
        let view = ConstrainedDevice::constrain(&physical, id)
            .unwrap_or_else(|e| panic!("R9700 card covers {id}: {e}"));
        for gate in [
            view.check_official_claim(),
            view.effective().check_official_claim(),
            view.provenance().check_official_claim(),
        ] {
            match gate {
                Err(IrError::SpoofQualificationRefused {
                    profile,
                    target,
                    disclaimer,
                }) => {
                    assert_eq!(profile, id.stable_id());
                    assert!(target.ends_with(" (SPOOF)"), "{target}");
                    assert_eq!(target, spoof_lookup(id).target_label());
                    assert!(!disclaimer.is_empty());
                }
                other => panic!("spoof result must refuse official use, got: {other:?}"),
            }
        }
    }
}
