// SPDX-License-Identifier: Apache-2.0
//! Shared types, error handling, ids, hashing, and byte-size helpers for R9V (Spec 14 §2).
//!
//! Architecture and repository coding standards are defined in `CONVENTIONS.md`.

pub mod bytes;
pub mod error;
pub mod hash;
pub mod ids;
pub mod rng;

pub use bytes::{
    format_byte_size, parse_byte_size, ByteSize, ByteSizeError, GIB, KIB, MIB, PIB, TIB,
};
pub use error::{R9vError, Result};
pub use hash::{
    xxh3_128, xxh3_128_with_seed, xxh3_64, xxh3_64_with_seed, Xxh3Hasher, Xxh3Hasher128,
};
pub use ids::{ReqId, SeqId, StepId};
pub use rng::SeededRng;
