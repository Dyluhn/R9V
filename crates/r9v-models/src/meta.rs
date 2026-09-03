// SPDX-License-Identifier: Apache-2.0
//! Typed GGUF metadata lookup interface without container dependency (Spec 8 §4, §10; card A1.3).
//!
//! Model definitions consume metadata exclusively through this trait.
//! Implementations may read directly from parsed GGUF container headers
//! or synthetic test fixtures without `r9v-models` depending on `r9v-format`.

use std::collections::BTreeMap;

use crate::error::ModelsError;

/// Typed GGUF metadata lookup trait (Spec 8 §4, §10; card A1.3).
///
/// Model definitions consume metadata exclusively through this trait.
/// Implementations may read directly from parsed GGUF container headers
/// or synthetic test fixtures without `r9v-models` depending on `r9v-format`.
// DECISION(A1.3): GgufMeta returns borrowed &str for string lookups and owned Vec for arrays to avoid allocations on scalar lookups while keeping array access safe; rejected returning an intermediate dynamic Value enum. Spec 8 §4, §10.
pub trait GgufMeta {
    /// Returns true if `key` exists in metadata.
    fn has(&self, key: &str) -> bool;

    /// Reads a string value by key.
    fn str(&self, key: &str) -> Result<&str, ModelsError>;

    /// Reads an unsigned 32-bit integer by key.
    fn u32(&self, key: &str) -> Result<u32, ModelsError>;

    /// Reads an unsigned 64-bit integer by key.
    fn u64(&self, key: &str) -> Result<u64, ModelsError>;

    /// Reads a signed 32-bit integer by key.
    fn i32(&self, key: &str) -> Result<i32, ModelsError>;

    /// Reads a 32-bit float by key.
    fn f32(&self, key: &str) -> Result<f32, ModelsError>;

    /// Reads a boolean by key.
    fn bool(&self, key: &str) -> Result<bool, ModelsError>;

    /// Reads a list of strings by key.
    fn str_array(&self, key: &str) -> Result<Vec<String>, ModelsError>;

    /// Reads a list of u32 values by key.
    fn u32_array(&self, key: &str) -> Result<Vec<u32>, ModelsError>;

    /// Reads an optional string value.
    fn get_str(&self, key: &str) -> Result<Option<&str>, ModelsError> {
        if self.has(key) {
            self.str(key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Reads an optional u32 value.
    fn get_u32(&self, key: &str) -> Result<Option<u32>, ModelsError> {
        if self.has(key) {
            self.u32(key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Reads an optional f32 value.
    fn get_f32(&self, key: &str) -> Result<Option<f32>, ModelsError> {
        if self.has(key) {
            self.f32(key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Reads an optional bool value.
    fn get_bool(&self, key: &str) -> Result<Option<bool>, ModelsError> {
        if self.has(key) {
            self.bool(key).map(Some)
        } else {
            Ok(None)
        }
    }
}

/// Dynamic value stored in [`SyntheticGgufMeta`] (Spec 8 §4; card A1.3).
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    /// String metadata value.
    Str(String),
    /// Unsigned 32-bit integer metadata value.
    U32(u32),
    /// Unsigned 64-bit integer metadata value.
    U64(u64),
    /// Signed 32-bit integer metadata value.
    I32(i32),
    /// 32-bit float metadata value.
    F32(f32),
    /// Boolean metadata value.
    Bool(bool),
    /// List of string metadata values.
    StrArray(Vec<String>),
    /// List of u32 metadata values.
    U32Array(Vec<u32>),
}

/// In-memory synthetic GGUF metadata implementation for testing and fixtures (Spec 8 §4, §8; card A1.3).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SyntheticGgufMeta {
    entries: BTreeMap<String, MetaValue>,
}

impl SyntheticGgufMeta {
    /// Creates an empty synthetic metadata container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a string key-value entry.
    pub fn insert_str(&mut self, key: impl Into<String>, val: impl Into<String>) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::Str(val.into()));
        self
    }

    /// Inserts an unsigned 32-bit integer key-value entry.
    pub fn insert_u32(&mut self, key: impl Into<String>, val: u32) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::U32(val));
        self
    }

    /// Inserts an unsigned 64-bit integer key-value entry.
    pub fn insert_u64(&mut self, key: impl Into<String>, val: u64) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::U64(val));
        self
    }

    /// Inserts a signed 32-bit integer key-value entry.
    pub fn insert_i32(&mut self, key: impl Into<String>, val: i32) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::I32(val));
        self
    }

    /// Inserts a 32-bit floating point key-value entry.
    pub fn insert_f32(&mut self, key: impl Into<String>, val: f32) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::F32(val));
        self
    }

    /// Inserts a boolean key-value entry.
    pub fn insert_bool(&mut self, key: impl Into<String>, val: bool) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::Bool(val));
        self
    }

    /// Inserts an array of string values.
    pub fn insert_str_array(&mut self, key: impl Into<String>, val: Vec<String>) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::StrArray(val));
        self
    }

    /// Inserts an array of u32 values.
    pub fn insert_u32_array(&mut self, key: impl Into<String>, val: Vec<u32>) -> &mut Self {
        self.entries.insert(key.into(), MetaValue::U32Array(val));
        self
    }

    /// Removes an entry by key if present.
    pub fn remove(&mut self, key: &str) -> Option<MetaValue> {
        self.entries.remove(key)
    }
}

impl GgufMeta for SyntheticGgufMeta {
    fn has(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    fn str(&self, key: &str) -> Result<&str, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::Str(s)) => Ok(s.as_str()),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "string",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "string",
            }),
        }
    }

    fn u32(&self, key: &str) -> Result<u32, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::U32(v)) => Ok(*v),
            Some(MetaValue::U64(v)) => {
                u32::try_from(*v).map_err(|_| ModelsError::MetaTypeMismatch {
                    key: key.to_string(),
                    expected: "u32",
                    found: format!("u64({v}) out of u32 range"),
                })
            }
            Some(MetaValue::I32(v)) => {
                u32::try_from(*v).map_err(|_| ModelsError::MetaTypeMismatch {
                    key: key.to_string(),
                    expected: "u32",
                    found: format!("i32({v}) negative"),
                })
            }
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "u32",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "u32",
            }),
        }
    }

    fn u64(&self, key: &str) -> Result<u64, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::U64(v)) => Ok(*v),
            Some(MetaValue::U32(v)) => Ok(*v as u64),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "u64",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "u64",
            }),
        }
    }

    fn i32(&self, key: &str) -> Result<i32, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::I32(v)) => Ok(*v),
            Some(MetaValue::U32(v)) => {
                i32::try_from(*v).map_err(|_| ModelsError::MetaTypeMismatch {
                    key: key.to_string(),
                    expected: "i32",
                    found: format!("u32({v}) out of i32 range"),
                })
            }
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "i32",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "i32",
            }),
        }
    }

    fn f32(&self, key: &str) -> Result<f32, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::F32(v)) => Ok(*v),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "f32",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "f32",
            }),
        }
    }

    fn bool(&self, key: &str) -> Result<bool, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::Bool(v)) => Ok(*v),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "bool",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "bool",
            }),
        }
    }

    fn str_array(&self, key: &str) -> Result<Vec<String>, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::StrArray(v)) => Ok(v.clone()),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "array of string",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "array of string",
            }),
        }
    }

    fn u32_array(&self, key: &str) -> Result<Vec<u32>, ModelsError> {
        match self.entries.get(key) {
            Some(MetaValue::U32Array(v)) => Ok(v.clone()),
            Some(other) => Err(ModelsError::MetaTypeMismatch {
                key: key.to_string(),
                expected: "array of u32",
                found: format!("{other:?}"),
            }),
            None => Err(ModelsError::MissingMetaKey {
                key: key.to_string(),
                expected_type: "array of u32",
            }),
        }
    }
}
