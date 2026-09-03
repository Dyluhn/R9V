//! Public API-shape checks for card A0.3 (Spec 12 §2, §4–5).

use r9v_config::{
    section, setting, Auto, CacheDtype, ConfigError, EffectiveConfig, GraphMode, IoMode, LogLevel,
    Mutability, ProfileMode, ProposerKind, SettingSpec, Source, SourcedValue, WarmupBuckets,
};

#[setting]
fn setting_macro_is_reexported() {}

#[section("fixture", doc = "External macro expansion fixture.")]
struct FixtureSection {
    #[setting(
        doc = "Fixture flag.",
        default = "false",
        mutable = Runtime,
        since = 1
    )]
    flag: bool,
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn public_schema_surface_is_send_sync() {
    assert_send_sync::<Auto<u32>>();
    assert_send_sync::<EffectiveConfig>();
    assert_send_sync::<Source>();
    assert_send_sync::<SourcedValue>();
    assert_send_sync::<SettingSpec>();
    assert_send_sync::<ConfigError>();
    assert_send_sync::<WarmupBuckets>();
    assert_send_sync::<IoMode>();
    assert_send_sync::<CacheDtype>();
    assert_send_sync::<GraphMode>();
    assert_send_sync::<ProposerKind>();
    assert_send_sync::<ProfileMode>();
    assert_send_sync::<LogLevel>();
}

#[test]
fn section_and_setting_macros_expand_for_external_users() {
    setting_macro_is_reexported();
    assert_eq!(FixtureSection::SECTION, "fixture");
    assert_eq!(
        FixtureSection::SECTION_DOC,
        "External macro expansion fixture."
    );
    assert_eq!(FixtureSection::len(), 1);
    let spec = FixtureSection::setting("fixture.flag").expect("fixture field is declared");
    assert_eq!(spec.key, "fixture.flag");
    assert_eq!(spec.mutability, Mutability::Runtime);
    let fixture = FixtureSection { flag: false };
    assert!(!fixture.flag);
}

#[test]
fn closed_setting_values_have_stable_spellings() {
    assert_eq!(IoMode::Direct.as_str(), "direct");
    assert_eq!(CacheDtype::E4m3.as_str(), "e4m3");
    assert_eq!(GraphMode::HipGraph.as_str(), "hipgraph");
    assert_eq!(ProposerKind::Mtp.as_str(), "mtp");
    assert_eq!(ProfileMode::Kernel.as_str(), "kernel");
    assert_eq!(LogLevel::Warn.as_str(), "warn");
}
