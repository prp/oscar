//! The runtime's public surface, one binary: what serve renders and parses, the store and its download, and the caches.

mod chat_template;
mod download;
#[path = "../../src/model_store/fixtures.rs"]
mod helpers;
mod kimi_k2_split;
mod prefix_cache;
mod rewrite;
mod tool_calls;
