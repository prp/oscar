//! The event lane's encoding side, on the forward thread; the resolver's side is `engine::event_lane`.

use super::*;
use crate::engine::event_lane::{gen_byte, gen_word, Block, Span, Tap};

/// Until committed less completed command buffers is at most [`ENCODE_AHEAD_CBS`]; the queue bounds count, not bytes.
pub(super) fn wait_encode_ahead(metal_device: &candle_core::MetalDevice) {
    let max_ahead = ENCODE_AHEAD_CBS;
    use objc2_metal::MTLSharedEvent as _;
    static PAIR: std::sync::OnceLock<(
        std::sync::Arc<std::sync::atomic::AtomicU64>,
        crate::engine::event_lane::SharedEventHandle,
    )> = std::sync::OnceLock::new();
    let (committed, completed) = PAIR.get_or_init(|| {
        (
            metal_device.completion_counter().expect("completion_counter"),
            crate::engine::event_lane::SharedEventHandle(metal_device.completion_event().expect("completion_event")),
        )
    });
    while committed.load(Ordering::Acquire).saturating_sub(completed.0.signaledValue()) > max_ahead {
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

impl OffloadDispatcher {
    /// No GPU sync: the resolver fills the rows and signals the kernel's event; `verified` holds a recording's ids.
    pub(super) fn forward_paged(&self, call: Call<'_>, verified: Option<Routing>) -> Result<Finalized> {
        let (layer, layer_idx, num_experts) = (call.layer, call.idx, call.shape.num_experts);
        let metal_device = &call.device;
        let pools = BankPools::new(metal_device, call.banks);
        let rows = &self.ep.scratch.rows;
        let ep = self.ep.state.get().expect("the event lane state is built at load");
        ep.check_resolver()?;
        // Each layer, or a bank's pins grow across a token and jam the pool once `topk × layers` exceeds it.
        ep.release_completed_pins(metal_device)?;

        let wait_value = ep.next_wait_value();
        let mut theta_span = None;
        let mut _keepalive: Vec<Tensor> = Vec::new();
        let xs_for_moe = &call.xs;
        let (topk_ids_u32, topk_weights_f32) = match &verified {
            Some(r) => (r.ids.clone(), r.weights.clone()),
            None => {
                let router_logits = layer.gate_forward(xs_for_moe)?;
                let (ids, weights) = layer.route_topk(&router_logits)?;
                // θ only on a one-token decode layer, as the resolver gates.
                let theta = ep.theta_on() && call.tokens == 1 && weights.is_contiguous();
                if theta {
                    crate::engine::approx::assert_softmax_family(
                        layer.gate_selection_bias().is_some(),
                        layer.norm_topk_prob(),
                    );
                }
                // The generation is baked in at encode, so a timed-out straggler cannot pass for a later dispatch.
                let tap_on = ep.layout.early_tap_words > 0 && self.router_tables.get().is_some() && call.tokens == 1;
                let look_b = ep.layout.gpu_pred_width;
                let pred_scores = (look_b > 0 && layer_idx + 1 < self.layout.block_count)
                    .then(|| self.pred_gate(layer_idx + 1))
                    .flatten()
                    .and_then(|(gate, bias)| {
                        crate::engine::router::selection_scores(&gate, bias.as_ref(), xs_for_moe)
                            .inspect_err(|e| {
                                tracing::error!(target: "oscar::oracle",
                                    "event lane: lookahead scores failed at layer {layer_idx}: {e} - prediction skipped",
                                )
                            })
                            .ok()
                    });
                let word = |layer, block| ep.layout.span(layer, block).word;
                _keepalive = oscar_kernels::metal_ep_publish(
                    &ids,
                    word(layer_idx, Block::Ids),
                    gen_word(wait_value, true),
                    pred_scores.as_ref(),
                    tap_on.then_some(xs_for_moe),
                    ep.mailbox_buffer(),
                    num_experts,
                    look_b,
                    word(layer_idx + 1, Block::Pred),
                    gen_word(wait_value, false),
                    word(layer_idx, Block::Tap(Tap::Early)),
                    gen_word(wait_value, false),
                )?;
                if theta {
                    let span = ep.layout.span(layer_idx, Block::Theta);
                    _keepalive.push(encode_theta_payload(
                        metal_device,
                        ep,
                        span,
                        wait_value,
                        &router_logits,
                        &weights,
                    )?);
                    theta_span = Some(span);
                }
                (ids, weights)
            }
        };
        // The blits commit now (a GPU write is visible to the CPU only at its buffer's completion) and the chain
        // rides the fresh buffer this flush opens, under the pool's count bound: `chain_gen` names it.
        metal_device.flush_commands()?;
        // After the publish and before the send: everything committed completes without this thread, so no cycle.
        wait_encode_ahead(metal_device);
        let chain_gen = ep.commit_generation() + 1;
        let ids = match verified {
            Some(r) => RoutingIds::Verified(r.host),
            None => RoutingIds::Mailbox,
        };
        let (target_tick, call_count) = (call.tick, call.call_count);
        let decode_token = self.decode_tokens.load(Ordering::Relaxed).saturating_sub(1);
        let d = crate::engine::event_lane::WorkItem {
            layer_idx,
            target_tick,
            wait_value,
            call_count,
            ids,
            decode_token,
            chain_gen,
        };
        ep.send_work(d)?;
        metal_device.set_pending_wait_event(ep.routing_event(), wait_value)?;
        if let Some(span) = theta_span {
            encode_theta_weights_writeback(metal_device, ep, span, &topk_weights_f32)?;
        }
        let rows = rows.layer_rows(metal_device, layer_idx)?;
        self.encode_layer(call, &pools, &rows, &topk_ids_u32, &topk_weights_f32, true)
    }
}

/// Between its gen bytes (gen-begin only for a double block), an encoder each, ordered on the hazard-tracked mailbox.
pub(super) fn write_block(
    device: &candle_core::MetalDevice, ep: &crate::engine::event_lane::EventLaneState, span: Span, wait_value: u64,
    labels: [&str; 3], payload: impl FnOnce(&candle_metal_kernels::metal::BlitCommandEncoder),
) -> Result<()> {
    let encoder = |label: &str| -> Result<_> {
        let enc = device.blit_command_encoder()?;
        enc.set_label(label);
        crate::engine::eval::hooks().note_blit(label);
        Ok(enc)
    };
    let fill = |offset: usize, label: &str| -> Result<()> {
        let enc = encoder(label)?;
        enc.fill_buffer(ep.mailbox_buffer(), (offset, 1), gen_byte(wait_value));
        enc.end_encoding();
        Ok(())
    };
    if span.double {
        fill(span.gen_begin_offset(), labels[0])?;
    }
    let enc = encoder(labels[1])?;
    payload(&enc);
    enc.end_encoding();
    fill(span.gen_offset(), labels[2])
}

/// The scores returned must outlive the caller's `flush_commands`, so the pooled source cannot recycle early.
fn encode_theta_payload(
    device: &candle_core::MetalDevice, ep: &crate::engine::event_lane::EventLaneState, span: Span, wait_value: u64,
    router_logits: &Tensor, topk_weights_f32: &Tensor,
) -> Result<Tensor> {
    let scores_src = router_logits.to_dtype(DType::F32)?.contiguous()?;
    let (s, wt) = (Metal::of(&scores_src, "theta scores")?, Metal::of(topk_weights_f32, "theta weights")?);
    let (scores_bytes, weights_bytes) = theta_split(ep, span);
    let labels = ["oscar_theta_gen_begin", "oscar_theta_payload", "oscar_theta_gen"];
    write_block(device, ep, span, wait_value, labels, |enc| {
        enc.copy_from_buffer(&s.buf, s.off, ep.mailbox_buffer(), span.offset(), scores_bytes);
        enc.copy_from_buffer(&wt.buf, wt.off, ep.mailbox_buffer(), span.offset() + scores_bytes, weights_bytes);
    })?;
    Ok(scores_src)
}

fn theta_split(ep: &crate::engine::event_lane::EventLaneState, span: Span) -> (usize, usize) {
    let scores = ep.layout.num_experts * std::mem::size_of::<f32>();
    (scores, span.bytes() - scores)
}

/// Encoded after the wait is armed, so it copies back what the resolver left once it has signalled.
fn encode_theta_weights_writeback(
    device: &candle_core::MetalDevice, ep: &crate::engine::event_lane::EventLaneState, span: Span,
    topk_weights_f32: &Tensor,
) -> Result<()> {
    let wt = Metal::of(topk_weights_f32, "theta weights")?;
    let (scores_bytes, weights_bytes) = theta_split(ep, span);
    let wb = device.blit_command_encoder()?;
    wb.set_label("oscar_theta_weights_writeback");
    crate::engine::eval::hooks().note_blit("oscar_theta_weights_writeback");
    wb.copy_from_buffer(ep.mailbox_buffer(), span.offset() + scores_bytes, &wt.buf, wt.off, weights_bytes);
    wb.end_encoding();
    Ok(())
}
