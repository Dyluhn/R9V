// SPDX-License-Identifier: Apache-2.0
//! Typed constrained-device (spoof) foundation for planning
//! (Spec 1 App. A, Spec 5 §2, Spec 14 §3).
//!
//! A physical [`DeviceDescriptor`] always stays truthful: discovery populates
//! it, nothing here mutates it, and the stable [`DeviceIdentity`] travels
//! unchanged into every view. Planning against a smaller card than the bench
//! hardware uses an *effective* view instead: [`ConstrainedDevice`] derives an
//! [`EffectiveDeviceView`] whose VRAM and CU counts are reduced to a catalog
//! profile, and [`Provenance`] records whether a plan ran against physical
//! facts or a spoof target.
//!
//! The effective view is deliberately **not** a [`DeviceDescriptor`]: it
//! carries no `measured` performance block and no `p2p` links, so physical
//! measured performance can never travel as a spoof fact, and there is no
//! conversion back into a bare descriptor that could drop provenance. Every
//! planning number the view exposes is reachable only beside
//! [`Provenance::Spoof`], and any attempt to use a spoof result for official
//! qualification or a performance claim fails with the typed
//! [`IrError::SpoofQualificationRefused`](crate::IrError::SpoofQualificationRefused)
//! refusal via [`Provenance::check_official_claim`].
//!
//! The pre-queue [`PreQueueLaunchContract`] turns the reduced CU count into a
//! deterministic `ROC_GLOBAL_CU_MASK` assignment that the launcher applies,
//! and validates a caller-supplied value before HIP queue creation; library
//! code never writes the process environment.
//!
//! Cross-crate contract (spec 14 §3): this crate supplies planning data and
//! validation only. It performs no hard allocation enforcement: the CU mask
//! narrows CU *visibility* for a queue, never VRAM (the VRAM bound is a
//! planning budget the loader enforces by refusing oversized plans), and the
//! separate `r9v-hip` integration owns applying the assignment and calling
//! [`PreQueueLaunchContract::validate_process_env`] before queue creation.
//! That integration is not in this branch; nothing here claims otherwise.
//!
//! Initial catalog (data, not branches): the only profiles are the two
//! gfx1201 spoof targets. Execution dispatches on [`SpoofProfileId`], never
//! on product-name strings.

use std::fmt;
use std::str::FromStr;

use crate::{DeviceDescriptor, DeviceIdentity, IrError};

/// Maximum CUs expressible in one CU mask word (Spec 1 App. A: the mask is a
/// single 64-bit word; both initial profiles fit).
pub const MAX_MASK_CUS: u32 = 64;

/// Launcher environment variable the ROCr runtime reads to restrict the CUs
/// visible to a queue (Spec 14 §3: runtime interaction surface).
///
/// Library code produces assignments as data via
/// [`PreQueueLaunchContract::env_assignment`]; it never writes the process
/// environment itself.
pub const CU_MASK_ENV_NAME: &str = "ROC_GLOBAL_CU_MASK";

/// Grammar template reported when a mask value cannot be decoded at all.
const CU_MASK_GRAMMAR: &str = "`0x` followed by 1-16 lowercase hex digits (contiguous low bits)";

/// Closed set of spoof planning targets (initial catalog; new targets land by
/// adding a variant plus its catalog row, never by string matching).
///
/// Serializes by stable id, never by discriminant (CONVENTIONS.md §3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpoofProfileId {
    /// 16 GiB / 64-CU gfx1201 spoof target.
    Rx9070Xt,
    /// 16 GiB / 56-CU gfx1201 spoof target.
    Rx9070,
}

impl SpoofProfileId {
    /// Every catalog profile, in catalog order (deterministic iteration).
    pub const fn all() -> [Self; 2] {
        [Self::Rx9070Xt, Self::Rx9070]
    }

    /// Stable catalog id used for parsing, errors and receipts.
    pub const fn stable_id(self) -> &'static str {
        match self {
            Self::Rx9070Xt => "rx-9070-xt-spoof",
            Self::Rx9070 => "rx-9070-spoof",
        }
    }

    /// Parses a stable catalog id; untrusted input enters here
    /// (CONVENTIONS.md §1.5).
    pub fn parse(s: &str) -> Result<Self, IrError> {
        let trimmed = s.trim();
        for id in Self::all() {
            if trimmed == id.stable_id() {
                return Ok(id);
            }
        }
        Err(IrError::UnknownSpoofProfile {
            id: s.to_owned(),
            known: Self::all().iter().map(|id| id.stable_id()).collect(),
        })
    }
}

impl fmt::Display for SpoofProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.stable_id())
    }
}

impl FromStr for SpoofProfileId {
    type Err = IrError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// One spoof planning target: the ISA it constrains plus the resource bounds
/// planning must fit (Spec 1 App. A quantities, Spec 5 §2 planning use).
///
/// Bare marketing names appear nowhere in this API: the only public label is
/// [`Self::target_label`], which always carries the `(SPOOF)` qualifier, so no
/// API path yields an unqualified product name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpoofProfile {
    /// Catalog key.
    pub id: SpoofProfileId,
    /// ISA target the profile constrains (e.g. `"gfx1201"`).
    pub arch: &'static str,
    /// VRAM bound planning must fit, in bytes.
    pub vram_bytes: u64,
    /// CU bound planning must fit.
    pub cu_count: u32,
}

impl SpoofProfile {
    /// Display label for the spoof target, always qualified
    /// (e.g. `"RX 9070 XT (SPOOF)"`).
    // DECISION(spoof-foundation): qualified labels are per-row catalog data
    // rather than runtime `format!("{model} (SPOOF)")`; rejected runtime
    // formatting because `const` labels keep the qualifier present in every
    // build path with no allocation.
    pub const fn target_label(self) -> &'static str {
        match self.id {
            SpoofProfileId::Rx9070Xt => "RX 9070 XT (SPOOF)",
            SpoofProfileId::Rx9070 => "RX 9070 (SPOOF)",
        }
    }
}

/// Static spoof catalog: the only initial profiles (data rows, not branches).
///
/// Execution looks rows up by [`SpoofProfileId`]; no code path matches on a
/// product-name string.
pub static SPOOF_CATALOG: [SpoofProfile; 2] = [
    SpoofProfile {
        id: SpoofProfileId::Rx9070Xt,
        arch: "gfx1201",
        vram_bytes: 16 * 1024 * 1024 * 1024,
        cu_count: 64,
    },
    SpoofProfile {
        id: SpoofProfileId::Rx9070,
        arch: "gfx1201",
        vram_bytes: 16 * 1024 * 1024 * 1024,
        cu_count: 56,
    },
];

/// The static catalog slice, in catalog order.
pub fn spoof_catalog() -> &'static [SpoofProfile] {
    &SPOOF_CATALOG
}

/// Looks a catalog row up by id (dispatch on the enum, never on names).
///
/// # Panics
///
/// Never: every [`SpoofProfileId`] variant has a row, enforced by the
/// exhaustive match and covered by `catalog_covers_every_profile_id`.
pub fn spoof_lookup(id: SpoofProfileId) -> &'static SpoofProfile {
    match id {
        SpoofProfileId::Rx9070Xt => &SPOOF_CATALOG[0],
        SpoofProfileId::Rx9070 => &SPOOF_CATALOG[1],
    }
}

/// Where a plan's device numbers came from (Spec 5 §2 identity, Spec 11
/// receipts carry this).
///
/// `Physical` means the plan used discovered facts directly. `Spoof` means it
/// used a reduced [`ConstrainedDevice`] view: the physical identity is
/// preserved verbatim and the target is always qualified, so a spoof plan can
/// never be mistaken for an official product or performance qualification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Plan ran against discovered physical facts.
    Physical {
        /// Stable identity of the device the facts were discovered from.
        identity: DeviceIdentity,
    },
    /// Plan ran against a spoof-constrained view of a physical device.
    Spoof {
        /// Stable identity of the underlying physical device (unchanged).
        physical: DeviceIdentity,
        /// Catalog target the view was constrained to.
        profile: SpoofProfileId,
    },
}

impl Provenance {
    /// Returns `true` for a spoof-constrained plan.
    pub const fn is_spoof(&self) -> bool {
        matches!(self, Self::Spoof { .. })
    }

    /// Returns `true` for a plan against discovered physical facts.
    pub const fn is_physical(&self) -> bool {
        matches!(self, Self::Physical { .. })
    }

    /// The stable physical identity behind this plan, spoof or not.
    pub const fn physical_identity(&self) -> &DeviceIdentity {
        match self {
            Self::Physical { identity } => identity,
            Self::Spoof { physical, .. } => physical,
        }
    }

    /// The qualified spoof target label (`Some("MODEL (SPOOF)")`), or `None`
    /// for physical provenance, which names no target at all.
    ///
    /// There is intentionally no accessor returning a bare product name: the
    /// only name this type can utter is the qualified one.
    pub fn target_label(&self) -> Option<&'static str> {
        match self {
            Self::Physical { .. } => None,
            Self::Spoof { profile, .. } => Some(spoof_lookup(*profile).target_label()),
        }
    }

    /// States what this provenance must never be used as. Spoof views refuse
    /// official product and performance qualification structurally: the target
    /// label always carries `(SPOOF)`, the physical identity is preserved, and
    /// this disclaimer names the boundary in one place for receipts to quote.
    pub const fn qualification_disclaimer(&self) -> &'static str {
        match self {
            Self::Physical { .. } => {
                "Physical device identity: discovered facts, not a product qualification."
            }
            Self::Spoof { .. } => {
                "Spoof-constrained planning view: not an official product qualification and not a performance claim."
            }
        }
    }

    /// Gate for official product-qualification or performance-claim use.
    ///
    /// Physical provenance passes: discovered and doctor-measured facts
    /// legitimately back receipts. Spoof provenance always refuses with the
    /// typed [`IrError::SpoofQualificationRefused`](crate::IrError::SpoofQualificationRefused)
    /// error: a disclaimer string alone never authorizes such use, so the
    /// refusal is a value the caller must handle, not text it may ignore.
    pub fn check_official_claim(&self) -> Result<(), IrError> {
        match self {
            Self::Physical { .. } => Ok(()),
            Self::Spoof { profile, .. } => Err(IrError::SpoofQualificationRefused {
                profile: profile.stable_id(),
                target: spoof_lookup(*profile).target_label(),
                disclaimer: self.qualification_disclaimer(),
            }),
        }
    }
}

/// Renders a [`DeviceIdentity`] truthfully without inventing fields.
fn fmt_identity(f: &mut fmt::Formatter<'_>, identity: &DeviceIdentity) -> fmt::Result {
    match identity {
        DeviceIdentity::Cpu => f.write_str("cpu"),
        DeviceIdentity::Gpu { uuid, pci_bdf } => {
            f.write_str(pci_bdf)?;
            if let Some(bytes) = uuid {
                f.write_str(" uuid=")?;
                for b in bytes {
                    write!(f, "{b:02x}")?;
                }
            }
            Ok(())
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Physical { identity } => {
                f.write_str("physical ")?;
                fmt_identity(f, identity)
            }
            Self::Spoof { physical, profile } => {
                f.write_str(spoof_lookup(*profile).target_label())?;
                f.write_str(" on ")?;
                fmt_identity(f, physical)
            }
        }
    }
}

/// Effective planning view of a physical device reduced to a spoof profile
/// (Spec 1 App. A quantities, Spec 5 §2 planning use).
///
/// This is deliberately **not** a [`DeviceDescriptor`]. It carries the shared
/// ISA [`ArchDescriptor`](crate::ArchDescriptor), the unchanged physical
/// [`DeviceIdentity`], and the reduced CU/VRAM bounds planning must fit — and
/// nothing else. In particular it carries no `measured` performance block and
/// no `p2p` links, so physical measured performance can never travel as a
/// spoof fact, and there is no `From`/`Into` conversion into a bare descriptor
/// that could drop provenance. Planning numbers are reachable only beside the
/// [`Provenance::Spoof`] this view also carries.
// DECISION(spoof-foundation): a distinct view type rather than a cloned
// descriptor; rejected cloning `DeviceDescriptor` because the clone carried
// `measured` and `p2p` across, letting a spoofed number be mistaken for a
// measured fact, and because a bare descriptor return lets callers drop
// provenance silently.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveDeviceView {
    /// Shared ISA capabilities (never per-device quantities).
    arch: crate::ArchDescriptor,
    /// Stable identity of the underlying physical device (unchanged).
    identity: DeviceIdentity,
    /// CU bound planning must fit (the catalog profile count).
    cu_count: u32,
    /// VRAM bound planning must fit, in bytes (the catalog profile bound).
    vram_bytes: u64,
    /// Catalog target this view was constrained to.
    profile: SpoofProfileId,
    /// Spoof provenance preserving the physical identity.
    provenance: Provenance,
}

impl EffectiveDeviceView {
    /// Shared ISA capabilities (never per-device quantities).
    pub const fn arch(&self) -> &crate::ArchDescriptor {
        &self.arch
    }

    /// Stable identity of the underlying physical device (unchanged).
    pub const fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    /// CU bound planning must fit (the catalog profile count).
    pub const fn cu_count(&self) -> u32 {
        self.cu_count
    }

    /// VRAM bound planning must fit, in bytes (the catalog profile bound).
    pub const fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// The catalog target this view was constrained to.
    pub const fn profile(&self) -> SpoofProfileId {
        self.profile
    }

    /// The catalog row this view was constrained to.
    pub fn profile_data(&self) -> &'static SpoofProfile {
        spoof_lookup(self.profile)
    }

    /// Spoof provenance preserving the physical identity; always
    /// [`Provenance::Spoof`] by construction.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Gate for official product-qualification or performance-claim use over
    /// this view. Always refuses with the typed
    /// [`IrError::SpoofQualificationRefused`](crate::IrError::SpoofQualificationRefused)
    /// error; see [`Provenance::check_official_claim`].
    pub fn check_official_claim(&self) -> Result<(), IrError> {
        self.provenance.check_official_claim()
    }
}

/// Constrained-device pair: the truthful physical descriptor beside its
/// reduced planning view (Spec 1 App. A quantities, Spec 5 §2 planning use).
///
/// The physical descriptor is stored by value and never mutated. Callers that
/// need discovered facts use [`Self::physical`]; callers that plan use
/// [`Self::effective`], whose [`EffectiveDeviceView`] type keeps the two from
/// being confused.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstrainedDevice {
    /// Truthful discovered descriptor; immutable after construction.
    physical: DeviceDescriptor,
    /// Planning view: reduced CU/VRAM bounds beside spoof provenance.
    effective: EffectiveDeviceView,
    /// Catalog target this view was constrained to.
    profile: SpoofProfileId,
    /// Spoof provenance preserving the physical identity.
    provenance: Provenance,
}

impl ConstrainedDevice {
    /// Derives the effective planning view for `profile` from `physical`.
    ///
    /// Refuses (collecting every problem, CONVENTIONS.md §1.4) when the
    /// physical arch differs from the profile's, or when the physical device
    /// cannot cover the profile's CUs or VRAM: constraints only reduce, so a
    /// smaller card can never spoof a larger one.
    pub fn constrain(
        physical: &DeviceDescriptor,
        profile: SpoofProfileId,
    ) -> Result<Self, IrError> {
        let row = spoof_lookup(profile);
        let mut problems = Vec::new();
        if physical.arch.name != row.arch {
            problems.push(IrError::SpoofArchMismatch {
                profile: profile.stable_id(),
                required_arch: row.arch,
                physical_arch: physical.arch.name.clone(),
            });
        }
        if physical.facts.cu_count < row.cu_count {
            problems.push(IrError::SpoofInsufficientCus {
                profile: profile.stable_id(),
                required_cus: row.cu_count,
                physical_cus: physical.facts.cu_count,
                shortfall_cus: row.cu_count - physical.facts.cu_count,
            });
        }
        if physical.facts.vram_bytes < row.vram_bytes {
            problems.push(IrError::SpoofInsufficientVram {
                profile: profile.stable_id(),
                required_bytes: row.vram_bytes,
                physical_bytes: physical.facts.vram_bytes,
                shortfall_bytes: row.vram_bytes - physical.facts.vram_bytes,
            });
        }
        IrError::from_problems(problems)?;

        let provenance = Provenance::Spoof {
            physical: physical.facts.identity.clone(),
            profile,
        };
        let effective = EffectiveDeviceView {
            arch: physical.arch.clone(),
            identity: physical.facts.identity.clone(),
            cu_count: row.cu_count,
            vram_bytes: row.vram_bytes,
            profile,
            provenance: provenance.clone(),
        };
        Ok(Self {
            physical: physical.clone(),
            effective,
            profile,
            provenance,
        })
    }

    /// The truthful discovered descriptor (never reduced).
    pub fn physical(&self) -> &DeviceDescriptor {
        &self.physical
    }

    /// The planning view (reduced CU/VRAM bounds beside spoof provenance).
    pub const fn effective(&self) -> &EffectiveDeviceView {
        &self.effective
    }

    /// The catalog target this view was constrained to.
    pub const fn profile(&self) -> SpoofProfileId {
        self.profile
    }

    /// The catalog row this view was constrained to.
    pub fn profile_data(&self) -> &'static SpoofProfile {
        spoof_lookup(self.profile)
    }

    /// Spoof provenance preserving the physical identity.
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// Gate for official product-qualification or performance-claim use over
    /// this device. Always refuses with the typed
    /// [`IrError::SpoofQualificationRefused`](crate::IrError::SpoofQualificationRefused)
    /// error; see [`Provenance::check_official_claim`].
    pub fn check_official_claim(&self) -> Result<(), IrError> {
        self.provenance.check_official_claim()
    }

    /// CUs removed by the constraint (`physical - effective`, never negative).
    pub fn cu_reduction(&self) -> u32 {
        self.physical
            .facts
            .cu_count
            .saturating_sub(self.effective.cu_count)
    }

    /// VRAM bytes removed by the constraint (`physical - effective`).
    pub fn vram_reduction_bytes(&self) -> u64 {
        self.physical
            .facts
            .vram_bytes
            .saturating_sub(self.effective.vram_bytes)
    }
}

/// Single-word CU enable mask for a reduced-CU launch (Spec 14 §3).
///
/// Canonical R9V form: the lowest `N` bits set, rendered `0x`-prefixed
/// lowercase hex with no leading zeros (e.g. 56 CUs is `0xffffffffffffff`).
/// Lowest-bits-first is deterministic across runs and tiers: the same CU
/// count always yields the same mask string.
// DECISION(spoof-foundation): lowest-N-bits mask; rejected strided or hashed
// CU subsets because only a fixed-order prefix keeps mask derivation a pure
// function of the CU count (standards §2.6 determinism).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CuMask(u64);

impl CuMask {
    /// Builds the mask enabling the lowest `cus` CUs.
    pub fn for_cu_count(cus: u32) -> Result<Self, IrError> {
        if cus == 0 || cus > MAX_MASK_CUS {
            return Err(IrError::InvalidCuCount {
                cus,
                max_supported: MAX_MASK_CUS,
            });
        }
        let bits = if cus == MAX_MASK_CUS {
            u64::MAX
        } else {
            (1u64 << cus) - 1
        };
        Ok(Self(bits))
    }

    /// Raw mask word.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Number of enabled CUs (set-bit count).
    pub fn cu_count(self) -> u32 {
        self.0.count_ones()
    }

    /// Canonical `ROC_GLOBAL_CU_MASK` value for this mask.
    pub fn to_env_value(self) -> String {
        format!("0x{:x}", self.0)
    }

    /// Parses a candidate mask, accepting only canonical form: `0x` prefix,
    /// 1–16 lowercase hex digits, no leading zeros, nonzero, contiguous low
    /// bits. Untrusted launcher input enters here (CONVENTIONS.md §1.5).
    pub fn parse(s: &str) -> Result<Self, IrError> {
        let invalid = |details: &str, expected: String| IrError::InvalidCuMask {
            input: s.to_owned(),
            details: details.to_owned(),
            expected,
        };
        let grammar_hint = || CU_MASK_GRAMMAR.to_owned();
        let digits = s
            .strip_prefix("0x")
            .ok_or_else(|| invalid("mask must start with `0x`", grammar_hint()))?;
        if digits.is_empty() || digits.len() > 16 {
            return Err(invalid(
                "mask must carry 1-16 hex digits after `0x`",
                grammar_hint(),
            ));
        }
        if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid("mask digits must be hexadecimal", grammar_hint()));
        }
        let value = u64::from_str_radix(digits, 16)
            .map_err(|_| invalid("mask digits do not decode as u64 hex", grammar_hint()))?;
        if value == 0 {
            return Err(invalid(
                "mask must enable at least one CU",
                "a nonzero mask such as `0x1`".to_owned(),
            ));
        }
        let canonical = format!("0x{value:x}");
        if s != canonical {
            return Err(invalid("mask is not canonical lowercase hex", canonical));
        }
        let contiguous = value == u64::MAX || (value & (value + 1)) == 0;
        if !contiguous {
            let fixed = Self(lowest_bits(value.count_ones()));
            return Err(invalid(
                "mask must enable the lowest N CUs contiguously",
                fixed.to_env_value(),
            ));
        }
        Ok(Self(value))
    }
}

/// Lowest-`n`-bits word; `n` is a live popcount (`1..=64`), so the shifts
/// below cannot overflow.
fn lowest_bits(n: u32) -> u64 {
    if n >= MAX_MASK_CUS {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

impl fmt::Display for CuMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_env_value())
    }
}

impl FromStr for CuMask {
    type Err = IrError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Pre-queue launch contract for a spoof-constrained device (Spec 14 §3).
///
/// Pure data: it records which profile a queue was planned for, how many CUs
/// the physical device has versus how many the queue may use, and — only when
/// the plan constrains CUs below the physical count — the exact
/// `ROC_GLOBAL_CU_MASK` assignment the launcher must apply before HIP queue
/// creation. Construction and validation are deterministic functions of the
/// [`ConstrainedDevice`]; this type has no method that writes the
/// environment, so library code cannot mutate launcher state.
///
/// Exact-CU hardware (physical CUs equal the profile bound) needs no mask:
/// [`Self::requires_mask`] is false and [`Self::env_assignment`] is `None`.
/// Reduced-CU targets use the deterministic lowest-N-bits mask
/// ([`CuMask::for_cu_count`]): the same CU count always yields the same mask
/// string across runs and tiers.
///
/// What this contract does **not** do: it never enforces VRAM allocation (the
/// mask narrows CU visibility only; the VRAM bound is a planning budget the
/// loader enforces by refusing oversized plans), and it never creates the HIP
/// queue itself. The separate `r9v-hip` integration must apply the assignment
/// and call [`Self::validate_process_env`] before queue creation; that
/// integration is not in this branch.
// DECISION(spoof-foundation): `Option` assignment rather than an empty-string
// sentinel; rejected a `""`-means-unset convention because an empty value is
// itself a malformed mask the runtime would misread, while `None` forces the
// launcher to branch explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreQueueLaunchContract {
    /// Catalog target the queue was planned for.
    profile: SpoofProfileId,
    /// CUs the physical device reports.
    physical_cus: u32,
    /// CUs the queue may use (the constrained effective count).
    effective_cus: u32,
    /// VRAM bytes the queue was planned for (the constrained bound).
    effective_vram_bytes: u64,
    /// Mask enabling exactly `effective_cus` CUs (meaningful only when
    /// [`Self::requires_mask`] holds; otherwise no assignment exists).
    mask: CuMask,
}

impl PreQueueLaunchContract {
    /// Builds the contract for a constrained device. Deterministic: the same
    /// device always yields the same mask string.
    pub fn for_constrained(device: &ConstrainedDevice) -> Self {
        let cus = device.effective().cu_count();
        // Internal invariant: `constrain` only succeeds for catalog profiles
        // whose CU counts are literals in 1..=64, so this cannot fail.
        let mask = CuMask::for_cu_count(cus)
            .expect("constrained effective CUs are catalog literals in 1..=64");
        Self {
            profile: device.profile(),
            physical_cus: device.physical().facts.cu_count,
            effective_cus: cus,
            effective_vram_bytes: device.effective().vram_bytes(),
            mask,
        }
    }

    /// Catalog target the queue was planned for.
    pub const fn profile(&self) -> SpoofProfileId {
        self.profile
    }

    /// CUs the physical device reports.
    pub const fn physical_cus(&self) -> u32 {
        self.physical_cus
    }

    /// CUs the queue may use.
    pub const fn effective_cus(&self) -> u32 {
        self.effective_cus
    }

    /// VRAM bytes the queue was planned for.
    pub const fn effective_vram_bytes(&self) -> u64 {
        self.effective_vram_bytes
    }

    /// The contract mask (meaningful only when [`Self::requires_mask`]
    /// holds; the launcher must branch on [`Self::env_assignment`]).
    pub const fn mask(&self) -> CuMask {
        self.mask
    }

    /// True when the plan constrains CUs below the physical count, so the
    /// launcher must assign a mask. False for exact-CU hardware, which needs
    /// no `ROC_GLOBAL_CU_MASK` assignment at all.
    pub const fn requires_mask(&self) -> bool {
        self.effective_cus < self.physical_cus
    }

    /// Environment variable the launcher assigns when [`Self::requires_mask`]
    /// holds.
    pub const fn env_name(&self) -> &'static str {
        CU_MASK_ENV_NAME
    }

    /// Canonical mask value the launcher must assign (defined even for
    /// exact-CU contracts, but with no assignment to carry it; prefer
    /// [`Self::env_assignment`]).
    pub fn env_value(&self) -> String {
        self.mask.to_env_value()
    }

    /// The `(name, value)` assignment the launcher applies before HIP queue
    /// creation, or `None` for exact-CU hardware, which needs no mask.
    /// Returned as data; applying it is the launcher's job, outside library
    /// code.
    pub fn env_assignment(&self) -> Option<(&'static str, String)> {
        if self.requires_mask() {
            Some((self.env_name(), self.env_value()))
        } else {
            None
        }
    }

    /// Validates the contract against a constrained device, collecting every
    /// mismatch (CONVENTIONS.md §1.4): profile, physical CU count, effective
    /// CU/VRAM bounds, and mask bit-count. This prevents reusing a same-profile
    /// exact-CU contract with a larger physical device that actually requires
    /// masking.
    pub fn validate_against(&self, device: &ConstrainedDevice) -> Result<(), IrError> {
        let mut problems = Vec::new();
        if self.profile != device.profile() {
            problems.push(IrError::SpoofProfileMismatch {
                contract_profile: self.profile.stable_id(),
                device_profile: device.profile().stable_id(),
            });
        }
        if self.physical_cus != device.physical().facts.cu_count {
            problems.push(IrError::SpoofLaunchContractMismatch {
                field: "physical_cus",
                contract: self.physical_cus.to_string(),
                device: device.physical().facts.cu_count.to_string(),
            });
        }
        if self.effective_cus != device.effective().cu_count() {
            problems.push(IrError::SpoofLaunchContractMismatch {
                field: "effective_cus",
                contract: self.effective_cus.to_string(),
                device: device.effective().cu_count().to_string(),
            });
        }
        if self.effective_vram_bytes != device.effective().vram_bytes() {
            problems.push(IrError::SpoofLaunchContractMismatch {
                field: "effective_vram_bytes",
                contract: self.effective_vram_bytes.to_string(),
                device: device.effective().vram_bytes().to_string(),
            });
        }
        if self.mask.cu_count() != device.effective().cu_count() {
            problems.push(IrError::CuMaskMismatch {
                mask: self.mask.to_env_value(),
                mask_cus: self.mask.cu_count(),
                effective_cus: device.effective().cu_count(),
            });
        }
        IrError::from_problems(problems)
    }

    /// Validates a caller-supplied `ROC_GLOBAL_CU_MASK` value (`None` = the
    /// variable is unset) against this contract, before HIP queue creation.
    ///
    /// Pure: takes the value as data, never reads or writes the process
    /// environment. Typed refusals for every wrong state:
    ///
    /// - exact-CU contract + `None` → `Ok` (no mask required);
    /// - exact-CU contract + `Some` → [`IrError::UnexpectedCuMask`];
    /// - reduced-CU contract + `None` → [`IrError::MissingCuMask`];
    /// - reduced-CU contract + malformed `Some` →
    ///   [`IrError::InvalidCuMask`] (via [`CuMask::parse`]);
    /// - reduced-CU contract + well-formed `Some` enabling the wrong CU count
    ///   → [`IrError::CuMaskMismatch`].
    ///
    /// Untrusted launcher input enters here (CONVENTIONS.md §1.5).
    pub fn validate_env_value(&self, value: Option<&str>) -> Result<(), IrError> {
        match value {
            None => {
                if self.requires_mask() {
                    Err(IrError::MissingCuMask {
                        profile: self.profile.stable_id(),
                        env_name: self.env_name(),
                        expected: self.env_value(),
                        effective_cus: self.effective_cus,
                    })
                } else {
                    Ok(())
                }
            }
            Some(input) => {
                if !self.requires_mask() {
                    return Err(IrError::UnexpectedCuMask {
                        profile: self.profile.stable_id(),
                        cus: self.effective_cus,
                        env_name: self.env_name(),
                        input: input.to_owned(),
                    });
                }
                let mask = CuMask::parse(input)?;
                if mask.cu_count() != self.effective_cus {
                    return Err(IrError::CuMaskMismatch {
                        mask: mask.to_env_value(),
                        mask_cus: mask.cu_count(),
                        effective_cus: self.effective_cus,
                    });
                }
                Ok(())
            }
        }
    }

    /// Reads the live `ROC_GLOBAL_CU_MASK` process value and validates it via
    /// [`Self::validate_env_value`]. Reads only: this method never writes the
    /// environment. The `r9v-hip` integration calls this before HIP queue
    /// creation on the spoof path; a non-UTF-8 value is a typed
    /// [`IrError::InvalidCuMask`] refusal, never a panic.
    pub fn validate_process_env(&self) -> Result<(), IrError> {
        match std::env::var(CU_MASK_ENV_NAME) {
            Ok(value) => self.validate_env_value(Some(&value)),
            Err(std::env::VarError::NotPresent) => self.validate_env_value(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(IrError::InvalidCuMask {
                input: "<non-UTF-8 value>".to_owned(),
                details: "mask value is not valid Unicode".to_owned(),
                expected: self.env_value(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchDescriptor, DeviceFacts, GraphCapture, Measured};

    const GIB: u64 = 1024 * 1024 * 1024;

    fn bench_gpu(cus: u32, vram_gib: u64) -> DeviceDescriptor {
        DeviceDescriptor {
            arch: ArchDescriptor::gfx1201(),
            facts: DeviceFacts {
                identity: DeviceIdentity::Gpu {
                    uuid: Some([0xabu8; 16]),
                    pci_bdf: "0000:03:00.0".to_owned(),
                },
                cu_count: cus,
                vram_bytes: vram_gib * GIB,
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

    #[test]
    fn catalog_has_initial_two_profiles_with_exact_bounds() {
        let catalog = spoof_catalog();
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog[0].id, SpoofProfileId::Rx9070Xt);
        assert_eq!(catalog[0].arch, "gfx1201");
        assert_eq!(catalog[0].vram_bytes, 16 * GIB);
        assert_eq!(catalog[0].cu_count, 64);
        assert_eq!(catalog[1].id, SpoofProfileId::Rx9070);
        assert_eq!(catalog[1].arch, "gfx1201");
        assert_eq!(catalog[1].vram_bytes, 16 * GIB);
        assert_eq!(catalog[1].cu_count, 56);
    }

    #[test]
    fn catalog_covers_every_profile_id() {
        for id in SpoofProfileId::all() {
            let row = spoof_lookup(id);
            assert_eq!(row.id, id);
            assert_eq!(row.target_label(), spoof_lookup(id).target_label());
        }
    }

    #[test]
    fn every_target_label_is_qualified_spoof() {
        for id in SpoofProfileId::all() {
            let label = spoof_lookup(id).target_label();
            assert!(
                label.ends_with(" (SPOOF)"),
                "unqualified target label: {label}"
            );
        }
        assert_eq!(
            spoof_lookup(SpoofProfileId::Rx9070Xt).target_label(),
            "RX 9070 XT (SPOOF)"
        );
        assert_eq!(
            spoof_lookup(SpoofProfileId::Rx9070).target_label(),
            "RX 9070 (SPOOF)"
        );
    }

    #[test]
    fn profile_id_parse_round_trips_stable_ids() {
        for id in SpoofProfileId::all() {
            assert_eq!(SpoofProfileId::parse(id.stable_id()), Ok(id));
            assert_eq!(id.to_string(), id.stable_id());
        }
    }

    #[test]
    fn profile_id_parse_rejects_unknown_with_known_ids() {
        let err = SpoofProfileId::parse("rx-9070-xt").unwrap_err();
        match err {
            IrError::UnknownSpoofProfile { id, known } => {
                assert_eq!(id, "rx-9070-xt");
                assert_eq!(known, vec!["rx-9070-xt-spoof", "rx-9070-spoof"]);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn spoof_display_always_qualifies_target_and_preserves_identity() {
        let physical = DeviceIdentity::Gpu {
            uuid: Some([0xabu8; 16]),
            pci_bdf: "0000:03:00.0".to_owned(),
        };
        let provenance = Provenance::Spoof {
            physical: physical.clone(),
            profile: SpoofProfileId::Rx9070Xt,
        };
        let rendered = provenance.to_string();
        assert!(rendered.starts_with("RX 9070 XT (SPOOF)"), "{rendered}");
        assert!(rendered.contains("0000:03:00.0"), "{rendered}");
        assert!(rendered.contains("abab"), "{rendered}");
        assert_eq!(provenance.physical_identity(), &physical);
        assert_eq!(provenance.target_label(), Some("RX 9070 XT (SPOOF)"));
        assert!(provenance.is_spoof());
        assert!(!provenance.is_physical());
        assert!(provenance.qualification_disclaimer().contains("not "));
    }

    #[test]
    fn physical_display_names_no_target() {
        let provenance = Provenance::Physical {
            identity: DeviceIdentity::Cpu,
        };
        assert_eq!(provenance.to_string(), "physical cpu");
        assert_eq!(provenance.target_label(), None);
        assert!(provenance.is_physical());
        assert!(!provenance.is_spoof());
    }

    #[test]
    fn constrain_reduces_only_vram_and_cus() {
        let physical = bench_gpu(96, 32);
        let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070)
            .expect("96 CUs / 32 GiB covers the 56-CU / 16 GiB profile");
        assert_eq!(view.effective().cu_count(), 56);
        assert_eq!(view.effective().vram_bytes(), 16 * GIB);
        assert_eq!(view.cu_reduction(), 40);
        assert_eq!(view.vram_reduction_bytes(), 16 * GIB);
        // ISA capabilities and physical identity carry over unchanged.
        assert_eq!(view.effective().arch(), &physical.arch);
        assert_eq!(view.effective().identity(), &physical.facts.identity);
        assert_eq!(view.profile(), SpoofProfileId::Rx9070);
        assert_eq!(view.provenance().target_label(), Some("RX 9070 (SPOOF)"));
        // Planning numbers always travel beside spoof provenance.
        assert!(view.effective().provenance().is_spoof());
        assert_eq!(
            view.effective().provenance().physical_identity(),
            &physical.facts.identity
        );
    }

    #[test]
    fn effective_view_drops_measured_and_p2p() {
        use crate::{P2pLink, P2pTransport};

        // A post-doctor physical descriptor: measured performance filled in
        // and a peer link present. None of it may become a spoof fact.
        let mut physical = bench_gpu(96, 32);
        physical.measured.mem_bw_gbps = Some(1800.0);
        physical.measured.dispatch_overhead_us = Some(3.0);
        physical.measured.h2d_gbps = Some(50.0);
        physical.p2p.push(P2pLink {
            peer_rank: 1,
            transport: P2pTransport::Direct,
            measured_gbps: Some(40.0),
        });
        let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070Xt)
            .expect("96 CUs / 32 GiB covers the XT profile");
        // The view type exposes no measured/p2p accessor by construction: the
        // only planning surface is bounds + identity + provenance, so this
        // test enumerates every accessor the view offers and asserts each is
        // either a bound, the shared ISA, the physical identity, or spoof
        // provenance carrying the full qualified label.
        let effective = view.effective();
        assert_eq!(effective.cu_count(), 64);
        assert_eq!(effective.vram_bytes(), 16 * GIB);
        assert_eq!(effective.arch(), &physical.arch);
        assert_eq!(effective.identity(), &physical.facts.identity);
        assert_eq!(effective.profile(), SpoofProfileId::Rx9070Xt);
        assert_eq!(
            effective.profile_data().target_label(),
            "RX 9070 XT (SPOOF)"
        );
        let rendered = effective.provenance().to_string();
        assert!(rendered.starts_with("RX 9070 XT (SPOOF)"), "{rendered}");
        // The physical side keeps its measured facts untouched for truthful
        // consumers; they simply never enter the view.
        assert_eq!(view.physical().measured.mem_bw_gbps, Some(1800.0));
        assert_eq!(view.physical().p2p.len(), 1);
    }

    #[test]
    fn qualification_gate_refuses_spoof_and_passes_physical() {
        let physical = bench_gpu(96, 32);
        for id in SpoofProfileId::all() {
            let view = ConstrainedDevice::constrain(&physical, id)
                .unwrap_or_else(|e| panic!("bench card covers {id}: {e}"));
            for gate in [
                view.provenance().check_official_claim(),
                view.check_official_claim(),
                view.effective().check_official_claim(),
            ] {
                match gate {
                    Err(IrError::SpoofQualificationRefused {
                        profile,
                        target,
                        disclaimer,
                    }) => {
                        assert_eq!(profile, id.stable_id());
                        assert!(target.ends_with(" (SPOOF)"), "{target}");
                        assert!(disclaimer.contains("not "), "{disclaimer}");
                    }
                    other => panic!("spoof gate must refuse, got: {other:?}"),
                }
            }
        }
        // Physical provenance backs ordinary receipts: the gate passes.
        let truthful = Provenance::Physical {
            identity: physical.facts.identity.clone(),
        };
        assert!(truthful.check_official_claim().is_ok());
    }

    #[test]
    fn constrain_leaves_physical_descriptor_untouched() {
        let physical = bench_gpu(96, 32);
        let snapshot = physical.clone();
        let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070Xt)
            .expect("covers the XT profile");
        assert_eq!(view.physical(), &snapshot);
        assert_eq!(physical, snapshot);
    }

    #[test]
    fn constrain_accepts_exact_size_match() {
        let physical = bench_gpu(64, 16);
        let view = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070Xt)
            .expect("exact match constrains cleanly");
        assert_eq!(view.cu_reduction(), 0);
        assert_eq!(view.vram_reduction_bytes(), 0);
    }

    #[test]
    fn constrain_rejects_wrong_arch_with_numbers() {
        let mut physical = bench_gpu(96, 32);
        physical.arch.name = "gfx1100".to_owned();
        let err = ConstrainedDevice::constrain(&physical, SpoofProfileId::Rx9070Xt).unwrap_err();
        match err {
            IrError::SpoofArchMismatch {
                profile,
                required_arch,
                physical_arch,
            } => {
                assert_eq!(profile, "rx-9070-xt-spoof");
                assert_eq!(required_arch, "gfx1201");
                assert_eq!(physical_arch, "gfx1100");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn constrain_rejects_smaller_card_with_every_shortfall() {
        let small = bench_gpu(32, 8);
        let err = ConstrainedDevice::constrain(&small, SpoofProfileId::Rx9070Xt).unwrap_err();
        match err {
            IrError::Multiple { problems } => {
                assert_eq!(problems.len(), 2);
                assert!(matches!(
                    &problems[0],
                    IrError::SpoofInsufficientCus {
                        required_cus: 64,
                        physical_cus: 32,
                        shortfall_cus: 32,
                        ..
                    }
                ));
                assert!(matches!(
                    &problems[1],
                    IrError::SpoofInsufficientVram {
                        required_bytes,
                        physical_bytes,
                        shortfall_bytes,
                        ..
                    } if *required_bytes == 16 * GIB
                        && *physical_bytes == 8 * GIB
                        && *shortfall_bytes == 8 * GIB
                ));
            }
            other => panic!("expected collect-all Multiple, got: {other:?}"),
        }
    }

    #[test]
    fn constrain_cpu_reports_arch_and_resource_problems_together() {
        let cpu = DeviceDescriptor::cpu();
        let err = ConstrainedDevice::constrain(&cpu, SpoofProfileId::Rx9070).unwrap_err();
        match err {
            IrError::Multiple { problems } => {
                assert_eq!(problems.len(), 3);
                assert!(matches!(&problems[0], IrError::SpoofArchMismatch { .. }));
                assert!(matches!(&problems[1], IrError::SpoofInsufficientCus { .. }));
                assert!(matches!(
                    &problems[2],
                    IrError::SpoofInsufficientVram { .. }
                ));
            }
            other => panic!("expected collect-all Multiple, got: {other:?}"),
        }
    }

    #[test]
    fn cu_mask_derivation_is_lowest_bits_deterministic() {
        assert_eq!(CuMask::for_cu_count(1).unwrap().to_env_value(), "0x1");
        assert_eq!(
            CuMask::for_cu_count(56).unwrap().to_env_value(),
            "0xffffffffffffff"
        );
        assert_eq!(
            CuMask::for_cu_count(64).unwrap().to_env_value(),
            "0xffffffffffffffff"
        );
        assert_eq!(
            CuMask::for_cu_count(56).unwrap(),
            CuMask::for_cu_count(56).unwrap()
        );
        assert_eq!(CuMask::for_cu_count(64).unwrap().cu_count(), 64);
    }

    #[test]
    fn cu_mask_rejects_out_of_range_counts_with_numbers() {
        for bad in [0, 65, 96, u32::MAX] {
            match CuMask::for_cu_count(bad).unwrap_err() {
                IrError::InvalidCuCount { cus, max_supported } => {
                    assert_eq!(cus, bad);
                    assert_eq!(max_supported, MAX_MASK_CUS);
                }
                other => panic!("wrong error for {bad}: {other:?}"),
            }
        }
    }

    #[test]
    fn cu_mask_parse_round_trips_canonical_values() {
        for cus in [1, 7, 32, 56, 63, 64] {
            let mask = CuMask::for_cu_count(cus).unwrap();
            let rendered = mask.to_env_value();
            assert_eq!(CuMask::parse(&rendered).unwrap(), mask);
            assert_eq!(rendered, mask.to_string());
        }
    }

    #[test]
    fn cu_mask_parse_rejects_noncanonical_forms() {
        for bad in [
            "",
            "0x",
            "0Xffffffffffffff",
            "0XFFFFFFFFFFFFFF",
            "0x00ffffffffffffff",
            "0x0",
            "0x5",
            "0x10",
            "0xFFFFFFFFFFFFFF",
            "ffffffffffffff",
            "0xzzzz",
            "0x1ffffffffffffffff",
        ] {
            assert!(
                CuMask::parse(bad).is_err(),
                "non-canonical mask accepted: {bad}"
            );
        }
        // Rejections carry the canonical fix.
        match CuMask::parse("0x00FF").unwrap_err() {
            IrError::InvalidCuMask {
                input,
                details,
                expected,
            } => {
                assert_eq!(input, "0x00FF");
                assert!(!details.is_empty());
                assert!(!expected.is_empty());
            }
            other => panic!("wrong error: {other:?}"),
        }
        match CuMask::parse("0x5").unwrap_err() {
            IrError::InvalidCuMask { expected, .. } => {
                assert_eq!(expected, "0x3");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn launch_contract_produces_env_assignment_as_data() {
        let view = ConstrainedDevice::constrain(&bench_gpu(96, 32), SpoofProfileId::Rx9070)
            .expect("covers the profile");
        let contract = PreQueueLaunchContract::for_constrained(&view);
        assert_eq!(contract.profile(), SpoofProfileId::Rx9070);
        assert_eq!(contract.physical_cus(), 96);
        assert_eq!(contract.effective_cus(), 56);
        assert_eq!(contract.effective_vram_bytes(), 16 * GIB);
        assert_eq!(contract.env_name(), "ROC_GLOBAL_CU_MASK");
        assert_eq!(contract.env_value(), "0xffffffffffffff");
        assert!(contract.requires_mask());
        assert_eq!(
            contract.env_assignment(),
            Some(("ROC_GLOBAL_CU_MASK", "0xffffffffffffff".to_owned()))
        );
    }

    #[test]
    fn exact_cu_contract_needs_no_mask() {
        // Physical CUs equal the profile bound: nothing to mask off.
        let view = ConstrainedDevice::constrain(&bench_gpu(64, 32), SpoofProfileId::Rx9070Xt)
            .expect("exact 64-CU match constrains cleanly");
        let contract = PreQueueLaunchContract::for_constrained(&view);
        assert_eq!(contract.physical_cus(), 64);
        assert_eq!(contract.effective_cus(), 64);
        assert!(!contract.requires_mask());
        assert_eq!(contract.env_assignment(), None);
        contract
            .validate_against(&view)
            .expect("exact contract validates against its own device");
        // Absent value passes; any supplied value is a typed refusal.
        assert!(contract.validate_env_value(None).is_ok());
        match contract
            .validate_env_value(Some("0xffffffffffffffff"))
            .unwrap_err()
        {
            IrError::UnexpectedCuMask {
                profile,
                cus,
                env_name,
                input,
            } => {
                assert_eq!(profile, "rx-9070-xt-spoof");
                assert_eq!(cus, 64);
                assert_eq!(env_name, "ROC_GLOBAL_CU_MASK");
                assert_eq!(input, "0xffffffffffffffff");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn reduced_cu_contract_validates_supplied_values_with_typed_errors() {
        let view = ConstrainedDevice::constrain(&bench_gpu(96, 32), SpoofProfileId::Rx9070)
            .expect("covers the profile");
        let contract = PreQueueLaunchContract::for_constrained(&view);
        assert!(contract.requires_mask());
        // Canonical 56-CU value passes.
        assert!(contract
            .validate_env_value(Some("0xffffffffffffff"))
            .is_ok());
        // Absent value refuses with the expected assignment attached.
        match contract.validate_env_value(None).unwrap_err() {
            IrError::MissingCuMask {
                profile,
                env_name,
                expected,
                effective_cus,
            } => {
                assert_eq!(profile, "rx-9070-spoof");
                assert_eq!(env_name, "ROC_GLOBAL_CU_MASK");
                assert_eq!(expected, "0xffffffffffffff");
                assert_eq!(effective_cus, 56);
            }
            other => panic!("wrong error: {other:?}"),
        }
        // Well-formed but wrong-count value refuses with both counts.
        match contract
            .validate_env_value(Some("0xffffffffffffffff"))
            .unwrap_err()
        {
            IrError::CuMaskMismatch {
                mask,
                mask_cus,
                effective_cus,
            } => {
                assert_eq!(mask, "0xffffffffffffffff");
                assert_eq!(mask_cus, 64);
                assert_eq!(effective_cus, 56);
            }
            other => panic!("wrong error: {other:?}"),
        }
        // Malformed values refuse as invalid masks, not mismatches.
        assert!(matches!(
            contract.validate_env_value(Some("0x5")),
            Err(IrError::InvalidCuMask { .. })
        ));
        assert!(matches!(
            contract.validate_env_value(Some("")),
            Err(IrError::InvalidCuMask { .. })
        ));
    }

    #[test]
    fn launch_contract_validates_against_matching_device() {
        let view = ConstrainedDevice::constrain(&bench_gpu(96, 32), SpoofProfileId::Rx9070Xt)
            .expect("covers the XT profile");
        let contract = PreQueueLaunchContract::for_constrained(&view);
        contract
            .validate_against(&view)
            .expect("matching contract validates");
        // Deterministic: rebuilt contracts compare equal.
        assert_eq!(contract, PreQueueLaunchContract::for_constrained(&view));
    }

    #[test]
    fn launch_contract_rejects_cross_profile_validation() {
        let xt = ConstrainedDevice::constrain(&bench_gpu(96, 32), SpoofProfileId::Rx9070Xt)
            .expect("covers the XT profile");
        let base = ConstrainedDevice::constrain(&bench_gpu(96, 32), SpoofProfileId::Rx9070)
            .expect("covers the base profile");
        let contract = PreQueueLaunchContract::for_constrained(&xt);
        let err = contract.validate_against(&base).unwrap_err();
        match err {
            IrError::Multiple { problems } => {
                assert!(matches!(&problems[0], IrError::SpoofProfileMismatch { .. }));
            }
            IrError::SpoofProfileMismatch {
                contract_profile,
                device_profile,
            } => {
                assert_eq!(contract_profile, "rx-9070-xt-spoof");
                assert_eq!(device_profile, "rx-9070-spoof");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }
}
