//! The routing read back to the host, which makes it authoritative, with its interventions and observers.

use super::*;

impl OffloadDispatcher {
    /// A census's spread at any width; at decode, an eval's override, which owns the routing outright, else θ.
    fn routing_intervention(
        &self, call: &Call<'_>, logits: &Tensor, weights: &Tensor, host: &[u32],
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        // A census's readback is whatever the buffer held; a spread over every expert makes the most waves.
        if oscar_offload::census::armed() {
            let n = self.layout.num_experts.max(1);
            let ids: Vec<u32> = (0..host.len()).map(|i| (i % n) as u32).collect();
            return Ok(Some((ids, vec![1.0f32; host.len()])));
        }
        if call.tokens != 1 {
            return Ok(None);
        }
        if let Some(overridden) = self.eval.routing_override(call.idx, call.call_count) {
            return Ok(overridden);
        }
        match crate::config::params().approx_theta {
            Some(theta) => theta_for_call(theta, call, logits, weights, host),
            None => Ok(None),
        }
    }

    /// The router's `topk_ids` as the authority: a predicted routing silently zeroes every live expert it missed.
    pub(super) fn read_back(&self, call: &Call<'_>) -> Result<Routing> {
        // Every routed call, so whichever phase reaches a layer first fills the gate book for both predictors.
        self.snapshot_gate(call.idx, call.layer);
        let t_router = std::time::Instant::now();
        let logits = call.layer.gate_forward(&call.xs)?;
        let (mut ids, mut weights) = call.layer.route_topk(&logits)?;
        let router_ns = t_router.elapsed().as_nanos() as u64;
        let t_readback = std::time::Instant::now();
        let mut host: Vec<u32> = ids.flatten_all()?.to_vec1::<u32>()?;
        if is_multi_token_forward(call.tokens) {
            let readback_ns = t_readback.elapsed().as_nanos() as u64;
            self.eval.prefill(PrefillEvent::Router { router_ns, readback_ns });
        }
        // Replaced once, before anything is pinned, fetched or executed, so the record banks the routing that ran.
        if let Some((new_ids, new_weights)) = self.routing_intervention(call, &logits, &weights, &host)? {
            ids = Tensor::from_slice(&new_ids, ids.shape(), ids.device())?;
            weights = Tensor::from_vec(new_weights, weights.shape(), weights.device())?;
            host = new_ids;
        }
        // The full router logits too, so every routing family is recoverable offline.
        if self.eval.records_routing() {
            let flat: Vec<f32> = weights.flatten_all()?.to_vec1::<f32>()?;
            let (idx, cc, n) = (call.idx, call.call_count, call.tokens as u32);
            self.eval.record_call(self, call.layer, idx, cc, n, &host, &flat, &logits)?;
        }
        let unique = dedup_u32_to_u16(&host, self.hot_budget());
        self.eval.check_routing(call.idx, call.call_count, call.tokens, &unique);
        // A census observes and never advances: its spread routing is no evidence and protects nothing.
        if !oscar_offload::census::armed() {
            self.protect_active_window(call, &unique);
        }
        Ok(Routing { ids, weights, host, unique })
    }

    /// The call's experts protected from eviction on all three banks; the routing stays the authority.
    fn protect_active_window(&self, call: &Call<'_>, unique: &[u16]) {
        let keys: Vec<LayerExpertKey> = unique.iter().map(|&e| (call.idx as u16, e)).collect();
        for bank in call.banks {
            bank.set_routed(keys.iter().copied());
        }
    }

    /// The two routes sum `down` in different orders, so the choice must depend on the layer alone.
    pub(super) fn takes_host_routing(&self, call: &Call<'_>, routing: &Routing) -> bool {
        let capacity = call.banks.iter().map(|b| b.capacity()).min().expect("three banks");
        call.tokens <= 1 && call.layer.has_fused_gateup() && routing.unique.len() <= capacity
    }

    /// Drains the resolver's pins, so a switch of lane cannot strand them across a prefill.
    pub(super) fn leave_gpu_routed_lane(&self, call: &Call<'_>) -> Result<()> {
        if let Some(ep) = self.ep.state.get() {
            ep.drain_unpins(&call.device)?;
        }
        Ok(())
    }
}

/// A decode call's substituted `(ids, weights)` against the live pool, or `None` where its routing stands.
fn theta_for_call(
    theta: f32, call: &Call<'_>, router_logits: &Tensor, topk_weights_f32: &Tensor, topk_ids_host: &[u32],
) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
    crate::engine::approx::assert_softmax_family(
        call.layer.gate_selection_bias().is_some(),
        call.layer.norm_topk_prob(),
    );
    let scores: Vec<f32> = router_logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let weights: Vec<f32> = topk_weights_f32.flatten_all()?.to_vec1::<f32>()?;
    let ids16: Vec<u16> = topk_ids_host.iter().map(|&e| e as u16).collect();
    let l16 = call.idx as u16;
    let is_res = |e: u16| call.banks.iter().all(|b| b.is_resident((l16, e)));
    Ok(crate::engine::approx::theta_substitute(theta, &ids16, &weights, &scores, &is_res)
        .map(|sub| (sub.exec.iter().map(|&e| e as u32).collect(), sub.weights)))
}
