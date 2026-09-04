//! Order-preserving JSON ingestion for template parity fixtures.
//!
//! Template contexts are insertion-ordered mappings (Jinja2 dict order):
//! `.items()` iteration must follow document order, but `serde_json::Map`
//! sorts keys without the `preserve_order` feature — which we must NOT
//! enable, since serde_json features unify across the workspace build and
//! other crates rely on the sorted shape. `serde_json`'s `MapAccess`
//! yields entries in document order regardless, so a manual
//! `Deserialize` impl collecting into a `Vec` preserves it with no
//! feature changes.

use r9v_loader::TemplateValue;
use serde::de::{MapAccess, SeqAccess, Visitor};
use std::fmt;

/// A JSON value with document-ordered objects.
#[derive(Debug, Clone)]
pub enum OrderedValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<OrderedValue>),
    Object(Vec<(String, OrderedValue)>),
}

impl OrderedValue {
    /// Parses one JSON document, preserving object key order.
    pub fn parse(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Object member lookup by key.
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            OrderedValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            OrderedValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            OrderedValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Self]> {
        match self {
            OrderedValue::Array(items) => Some(items),
            _ => None,
        }
    }
}

struct OrderedVisitor;

impl<'de> Visitor<'de> for OrderedVisitor {
    type Value = OrderedValue;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedValue::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(OrderedValue::Null)
    }

    fn visit_bool<E>(self, v: bool) -> Result<Self::Value, E> {
        Ok(OrderedValue::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E> {
        Ok(OrderedValue::Int(v))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if let Ok(i) = i64::try_from(v) {
            Ok(OrderedValue::Int(i))
        } else {
            Ok(OrderedValue::Float(v as f64))
        }
    }

    fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E> {
        Ok(OrderedValue::Float(v))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(OrderedValue::Str(v.to_owned()))
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
        Ok(OrderedValue::Str(v))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = seq.size_hint().map_or_else(Vec::new, Vec::with_capacity);
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(OrderedValue::Array(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = map.size_hint().map_or_else(Vec::new, Vec::with_capacity);
        while let Some((k, v)) = map.next_entry::<String, OrderedValue>()? {
            entries.push((k, v));
        }
        Ok(OrderedValue::Object(entries))
    }
}

impl<'de> serde::Deserialize<'de> for OrderedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(OrderedVisitor)
    }
}

/// Converts to a template value, preserving object key order.
pub fn to_template_value(v: &OrderedValue) -> TemplateValue {
    match v {
        OrderedValue::Null => TemplateValue::None,
        OrderedValue::Bool(b) => TemplateValue::Bool(*b),
        OrderedValue::Int(i) => TemplateValue::Int(*i),
        OrderedValue::Float(f) => TemplateValue::Float(*f),
        OrderedValue::Str(s) => TemplateValue::Str(s.clone()),
        OrderedValue::Array(items) => {
            TemplateValue::List(items.iter().map(to_template_value).collect())
        }
        OrderedValue::Object(entries) => TemplateValue::Dict(
            entries
                .iter()
                .map(|(k, v)| (k.clone(), to_template_value(v)))
                .collect(),
        ),
    }
}
