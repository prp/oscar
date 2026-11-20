//! The paged recurrent-state prefix cache: where it deposits, and a match resuming at the deepest stored ancestor.

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use mistralrs_core::kv_cache::RecurrentState;
use mistralrs_core::paged_attention::block_hash::{compute_block_hashes, BlockHash};
use mistralrs_core::paged_deposit_cuts;
use mistralrs_core::prefix_cacher::{usable_hit, PrefixCacheManagerV2};

fn snap(seqlen_offset: usize) -> RecurrentState {
    let t = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
    RecurrentState::from_layers(vec![(t.clone(), t)], seqlen_offset)
}

/// A distinct token stream per `salt`, hashed at block size 4.
fn hashes(salt: u32, tokens: usize) -> Vec<BlockHash> {
    let toks: Vec<u32> = (0..tokens as u32).map(|i| i.wrapping_mul(7) ^ salt).collect();
    compute_block_hashes(&toks, 4, &[], &[])
}

/// One device's cache: its cap is 8 states.
fn paged_cacher() -> PrefixCacheManagerV2 {
    PrefixCacheManagerV2::new(1, false, true, &Device::Cpu)
}

/// A state at `prefix`'s end, 4 tokens a block.
fn deposit(pc: &mut PrefixCacheManagerV2, prefix: &[BlockHash]) {
    pc.add_paged_recurrent_prefix(prefix.to_vec(), || Ok(snap(4 * prefix.len())), None).unwrap();
}

/// The depths of the states kept along `key`, shallow to deep.
fn depths(pc: &PrefixCacheManagerV2, key: &[BlockHash]) -> Vec<usize> {
    pc.paged_recurrent_ancestors(key).into_iter().map(|(d, _)| d).collect()
}

/// A recurrent entry is a point state: a snapshot past the divergence serves no partial match. Cuts align down to
/// the 32-token block, strictly inside the uncached span, in order; the resume boundary's floor is a cut, so every
/// prompt deposits within a block of where the next request resumes: a warm append, not a rewind.
#[test]
fn a_recurrent_entry_rewinds_only_to_a_boundary_before_the_divergence_and_a_match_resumes_at_the_deepest_one() {
    for (match_len, entry_len, request_len) in [(89, 100, 150), (90, 100, 90)] {
        let row = format!("match {match_len}, entry {entry_len}, request {request_len}");
        assert_eq!(usable_hit(true, Some(90), match_len, entry_len, request_len), None, "{row}");
        assert_eq!(usable_hit(true, Some(88), match_len, entry_len, request_len), Some(88), "{row}");
    }
    for (cached, end, plan, want) in [
        // A single-chunk prompt takes no cut.
        (0, 5961, vec![], vec![]),
        (0, 20, vec![], vec![]),
        // Two plan cuts aligning down to one dedup.
        (0, 15000, vec![5300, 12500, 12510], vec![5280, 12480]),
        (0, 15000, vec![5300, 10600], vec![5280, 10592]),
        (5952, 6100, vec![6080], vec![6080]),
        // A cut whose floor is not past the cached offset adds nothing.
        (5952, 6100, vec![5952], vec![]),
        // A cut at the prompt end aligns down inside.
        (0, 6100, vec![6100], vec![6080]),
        (0, 6080, vec![6080], vec![]),
    ] {
        let row = format!("cached {cached}, end {end}, plan {plan:?}");
        assert_eq!(paged_deposit_cuts(cached, end, 32, plan), want, "{row}");
    }
    // A match finds the stored state at its own depth, else the deepest ancestor below it, and none on another
    // stream; a decode-time deposit supersedes its depositor's last entry; durable cuts it does not name survive.
    let mut pc = paged_cacher();
    let (exact, deeper, stored, divergent) = (hashes(1, 40), hashes(2, 64), hashes(3, 40), hashes(4, 40));
    [&exact[..8], &deeper[..3], &deeper[..9], &stored[..8]].iter().for_each(|p| deposit(&mut pc, p));
    for (key, want) in [(&exact[..8], Some((8, 32))), (&deeper[..14], Some((9, 36))), (&divergent[..8], None)] {
        let got = pc.longest_paged_recurrent_ancestor(key).map(|(depth, snaps)| (depth, snaps.seqlen_offset()));
        assert_eq!(got, want, "{} blocks", key.len());
    }
    let mut pc = paged_cacher();
    let h = hashes(5, 64);
    deposit(&mut pc, &h[..2]);
    deposit(&mut pc, &h[..6]);
    pc.add_paged_recurrent_prefix(h[..9].to_vec(), || Ok(snap(36)), Some(&h[..6])).unwrap();
    // The superseded depth is gone: below the new tip, the durable cut answers.
    let mut deepest = |key: &[BlockHash]| pc.longest_paged_recurrent_ancestor(key).unwrap().0;
    assert_eq!((deepest(&h[..5]), deepest(&h[..16])), (2, 9));
}

/// Room is made before each take, so no take runs at the cap, and a rung the rule would evict at once is never taken;
/// pressure across conversations decays a stale one rung by rung, tip last, before the active one loses anything;
/// at the cap the coldest entry goes, and an ancestor hit refreshes.
#[test]
fn the_cap_keeps_the_tip_thins_the_ladder_geometrically_and_evicts_the_coldest() {
    struct Held(Arc<AtomicUsize>);
    impl Drop for Held {
        fn drop(&mut self) {
            self.0.fetch_sub(1, SeqCst);
        }
    }
    let live = Arc::new(AtomicUsize::new(0));
    let counted = |d: usize| {
        let live = Arc::clone(&live);
        move || {
            assert!(live.fetch_add(1, SeqCst) < 8, "the take at depth {d} ran with the cap's 8 states held");
            let t = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
            Ok(RecurrentState::new(None, vec![(t.clone(), t)], Box::new(Held(live))))
        }
    };
    let mut pc = paged_cacher();
    let h = hashes(40, 4 * 12);
    (1..=12).for_each(|d| pc.add_paged_recurrent_prefix(h[..d].to_vec(), counted(d), None).unwrap());
    pc.add_paged_recurrent_prefix(h[..6].to_vec(), || panic!("the rung the rule evicts at once was taken"), None)
        .unwrap();
    assert_eq!((depths(&pc, &h[..12]), live.load(SeqCst)), (vec![1, 5, 7, 8, 9, 10, 11, 12], 8));
    // The stale conversation.
    let mut pc = paged_cacher();
    let (a, b) = (hashes(50, 4 * 20), hashes(51, 4 * 8));
    for d in [2, 4, 6] {
        deposit(&mut pc, &b[..d]);
        deposit(&mut pc, &a[..d]);
    }
    // Three more turns on `a` push the map to the cap of 8 and one past: `b` lost its rung farthest behind the tip.
    [8, 10, 12].iter().for_each(|&d| deposit(&mut pc, &a[..d]));
    assert_eq!(depths(&pc, &b[..8]), [4, 6]);
    // Two more on `a`: `b` thins to its tip, then dies, while `a` keeps every rung.
    deposit(&mut pc, &a[..14]);
    assert_eq!(depths(&pc, &b[..8]).len(), 1);
    deposit(&mut pc, &a[..16]);
    assert_eq!((depths(&pc, &b[..8]).len(), depths(&pc, &a[..16]).len()), (0, 8));
    // The coldest: touch `a` so `b` is, then six more fill the cap of 8 and the ninth evicts exactly `b`.
    let mut pc = paged_cacher();
    let (a, b) = (hashes(10, 16), hashes(11, 16));
    deposit(&mut pc, &a[..4]);
    deposit(&mut pc, &b[..4]);
    assert!(pc.longest_paged_recurrent_ancestor(&a[..4]).is_some());
    (20..26).chain([30]).for_each(|salt| deposit(&mut pc, &hashes(salt, 16)[..4]));
    let mut kept = |key: &[BlockHash]| pc.longest_paged_recurrent_ancestor(key).is_some();
    assert_eq!((kept(&b[..4]), kept(&a[..4])), (false, true));
}
