// SPDX-License-Identifier: Apache-2.0
//! R9V Op IR core types (Spec 1 §2, App. A; card A1.1).
//!
//! This crate owns the API-bearing type surface the rest of the engine builds
//! against: dtypes, quant-scheme tags, tensors, batch metadata, state
//! handles, the arch descriptor and the IR version. Ops, graphs, sharding
//! tables and the numerics contract are owned by card A1.2 (Spec 1 §3–§6).
//!
//! Repository standards: `CONVENTIONS.md`; engineering bar:
//! `.agents/skills/r9v-engineering-standards`.

pub mod arch;
pub mod batch;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod quant;
pub mod state;
pub mod tensor;
pub mod version;

pub use arch::{
    ArchDescriptor, ArchFamily, GraphCapture, MatrixOp, Measured, P2pLink, P2pTransport, RelRate,
    ValuDot,
};
pub use batch::{BatchMeta, BatchMetaBuilder, Positions, TreeMask, BLOCK_TABLE_SENTINEL};
pub use dtype::DType;
pub use error::IrError;
pub use layout::LayoutId;
pub use quant::{QuantScheme, SchemeId};
pub use state::{StateHandle, StateKind};
pub use tensor::{Class, Dim, Placement, ShapeSymbol, ShardLayout, Tensor};
pub use version::IrVersion;
