//! What each architecture states about itself ([`Architecture`]), and the loaders' refusal ([`ensure`]).

use std::sync::Arc;

use candle_core::Result;
use mistralrs_core::gguf::GgufArch;

use crate::draft::DraftableModel;

pub(crate) fn ensure(ok: bool, msg: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        candle_core::bail!("{msg}")
    }
}

/// What an architecture states about itself, in its own file.
pub(crate) trait Architecture: GgufArch + Sized + 'static {
    /// It provisions the host router tables; one with a wide router may decline, and `OSCAR_LOOKAHEAD=host` then stops.
    const ROUTER_TABLES: bool;
    /// It offers the late tap, the residual entering a layer before its attention: the freshest a lookahead can read.
    const LATE_TAP: bool;
    /// Drafting reads the engine's rank-5 GQA pools, so no MLA or KDA architecture drafts.
    fn draftable(_model: Arc<Self>) -> Option<Arc<dyn DraftableModel>> {
        None
    }
}
