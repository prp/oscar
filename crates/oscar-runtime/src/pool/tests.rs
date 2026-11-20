use super::*;
use crate::pool::bank::tests::{borrowed, device, fill, temp_path, test_lock};

/// The paged lane's first layer is `layers - count`, where its per-token hooks must key: on 0 they never fire. The
/// banks iterate in the classes' order; `index()` addresses the format array, dense in `ALL`'s order.
#[test]
fn a_mixed_plan_seals_the_prefix_and_carries_the_tail_and_the_classes_sort_by_projection_then_format() {
    let mixed = carrier_of(16, 48);
    assert!(mixed[..32].iter().all(|&c| !c) && mixed[32..].iter().all(|&c| c), "the prefix seals, the tail carries");
    assert_eq!(mixed.iter().position(|&c| c), Some(32), "the lane's first layer is layers - count");
    assert!(carrier_of(48, 48)[0], "all-carrier: layer 0 carries");
    use crate::gpu::ProjectionKind::{Down, Gate, Up};
    use crate::pool::BankDtype::{Q3K, Q4K, Q6K};
    let mut classes = vec![(Down, Q6K), (Up, Q4K), (Down, Q3K), (Gate, Q4K), (Down, Q4K)];
    classes.sort();
    assert_eq!(classes, vec![(Gate, Q4K), (Up, Q4K), (Down, Q3K), (Down, Q4K), (Down, Q6K)]);
    assert!(ProjectionKind::ALL.into_iter().enumerate().all(|(i, kind)| kind.index() == i), "dense in ALL's order");
}

/// 3 layers of 3 experts, each extent its own byte; a segment, one stack.
fn pool_over_file() -> (std::path::PathBuf, ExpertPool) {
    let (layers, experts, bpe) = (3usize, 3usize, 4 * 16384usize);
    let path = temp_path("group");
    let content: Vec<u8> = (0..layers * 3 * experts).flat_map(|i| std::iter::repeat_n(i as u8 + 1, bpe)).collect();
    std::fs::write(&path, &content).expect("write the model file");
    let extent = |i: usize| ExpertExtent { shard_id: 0, file_offset: (i * bpe) as u64, byte_count: bpe };
    let projections = (0..layers)
        .flat_map(|l| ProjectionKind::ALL.into_iter().map(move |kind| (l, kind)))
        .map(|(l, kind)| {
            let extents = (0..experts).map(|e| extent((l * 3 + kind.index()) * experts + e)).collect();
            ((l, kind), ProjectionLayout { dtype: GgmlDType::Q4K, n: 1, k: 1, bytes_per_expert: bpe, experts: extents })
        })
        .collect();
    let layout = GgufRoutedLayout {
        block_count: layers,
        nextn_blocks: 0,
        num_experts: experts,
        experts_per_token: Some(2),
        projections,
        activation: Default::default(),
    };
    let pool = ExpertPool::build(&layout, std::slice::from_ref(&path), 1).expect("build");
    (path, pool)
}

/// The plan's shape at `hot_budget` slots a carrier, the last `carriers` layers carrying.
fn launch(pool: &ExpertPool, hot_budget: usize, carriers: usize) {
    struct Unscored;
    impl Oracle for Unscored {
        fn eviction_score(&self, _: LayerExpertKey, _: u32) -> u32 {
            u32::MAX
        }
    }
    let launch =
        Launch { hot_budget, pacer: SchedulePacer::Procedural(DecodePacer::new(3, 1)), oracle: Arc::new(Unscored) };
    pool.launch(hot_budget, carriers, launch).expect("launch");
}

/// The mover's half of the last move, settled at `tick`.
fn settle(pool: &ExpertPool, tick: u32) -> Result<Moves> {
    pool.tick.store(tick, Ordering::Relaxed);
    let settle = pool.settling.lock().unwrap().take().expect("a move owes its settle");
    pool.settle(&settle, || true)
}

/// Launched with layers 0 and 1 sealed and layer 2 carrying, then layer 1 shed: its segment retired and released.
fn shed_layer_1() -> (std::path::PathBuf, ExpertPool) {
    let (path, pool) = pool_over_file();
    launch(&pool, 3, 1);
    pool.tick.store(1, Ordering::Relaxed);
    assert_eq!(pool.move_to(1, &[false, true, true], 1).retired, 3, "one segment of each bank");
    assert_eq!(pool.housekeep(1, true), (3, 0, None), "every retired segment released");
    (path, pool)
}

/// A shed retires the unsealed layer's segment and keeps the carrier's; the regrow admits a fresh one, and the re-seal
/// reads the layer again, taking the experts the carriers hold from their copies.
#[test]
fn a_shed_retires_the_unsealed_layers_segment_and_a_re_seal_copies_what_the_carriers_hold_and_reads_the_rest() {
    let _g = test_lock();
    let (path, pool) = shed_layer_1();
    for (_, s) in pool.banks() {
        assert!(!s.segments()[1].present, "the unsealed layer's segment is released");
        let seg = s.segments()[2];
        assert_eq!((seg.present, seg.sealed, seg.retiring), (true, 0, false), "the carrier's segment stays");
        // The carrier's demand lands in the segment that stays.
        (0..3).for_each(|e| _ = fill(s, (2, e), 2));
    }
    pool.move_to(3, &[false, false, true], 2);
    let moves = settle(&pool, 3).expect("the re-seal reads the layer");
    assert_eq!((moves.admitted, moves.copied, moves.read), (3, 0, 9), "a fresh segment a bank, every expert read");
    assert!(pool.layer_sealed(1));
    let _ = std::fs::remove_file(&path);
    // The carriers holding two of layer 1's experts: copied in each bank, the third read.
    let (path, pool) = shed_layer_1();
    for (_, s) in pool.banks() {
        fill(s, (1, 0), 2);
        fill(s, (1, 1), 2);
    }
    pool.move_to(2, &[false, true, true], 2);
    let moves = settle(&pool, 3).expect("settle");
    assert_eq!(moves.admitted, 3, "a fresh segment a bank");
    pool.move_to(3, &[false, false, true], 3);
    let moves = settle(&pool, 4).expect("the third expert read");
    assert_eq!((moves.copied, moves.read), (6, 3), "the two the carriers hold copied in each bank, the third read");
    assert!(pool.layer_sealed(1));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_loans_are_laid_by_level_and_a_take_back_hands_back_only_what_is_under_the_bound() {
    use candle_metal_kernels::metal::{Fences, Use};
    use oscar_offload::segments::PAGE;
    use std::sync::atomic::{AtomicU64, AtomicUsize};
    let _g = test_lock();
    let (path, pool) = pool_over_file();
    let carriers = [true, true, true];
    launch(&pool, 1, 3);
    let device = device();
    let event = objc2_metal::MTLDevice::newSharedEvent(&*device).expect("a shared event");
    let fences = Fences::new(Arc::new(AtomicU64::new(0)), event, 2, Arc::new(vec![Arc::new(AtomicUsize::new(0))]));
    // Segments of three 4-page slots; a 96-page region, decode's bound 16.
    let (segment, mapped, floor) = (12 * PAGE, 96 * PAGE, 16 * PAGE);
    let region = oscar_offload::region::Region::take(&device, 0, mapped, fences, true).expect("a region");
    assert_eq!(pool.plan_loans(1, 3, &carriers, mapped, floor), 6, "two levels of a segment a bank");
    let classes: Vec<_> = pool.banks().map(|(c, _)| *c).collect();
    for (i, loan) in pool.loans().iter().enumerate() {
        let laid = (classes[i % 3], 5 - i / 3, mapped - (i + 1) * segment, segment);
        assert_eq!((loan.class, loan.seg, loan.offset, loan.len), laid);
        assert!(!pool.banks[&loan.class].segments()[loan.seg].borrowed, "laid, not lent");
    }
    let in_range = pool.plan_loans(1, 3, &carriers, mapped, mapped - 4 * segment);
    assert_eq!(in_range, 3, "whole levels inside the range, never under the floor");
    pool.plan_loans(1, 3, &carriers, mapped, floor);
    let held = region.object(PAGE, Use::Scratch, "held").expect("an object");
    assert!(pool.lend(&region, floor).is_err(), "a region with an object in hand lends nothing");
    drop(held);
    assert_eq!(pool.lend(&region, floor).expect("idle: lent"), 6);
    assert_eq!((region.bound(), region.levels().lent), (floor, mapped - floor));
    let base = region.lender().base.as_ptr() as usize;
    for (_, s) in pool.banks() {
        assert_eq!((s.capacity(), borrowed(s)), (9, vec![4, 5]));
        // Carriers fill past the own segment into the region, the first fills in the loan a take-back reaches last.
        (0..9).for_each(|i| _ = fill(s, (i / 3, i % 3), 1));
        for e in 0..3 {
            let slot = s.resident_slot((0, e)).expect("resident");
            let at = s.pool().ptr(slot) as usize - base;
            let top = (15..18).contains(&slot) && at >= mapped - 3 * segment;
            assert!(top, "slot {slot} stands at the top of the region's mapping");
        }
    }
    // A shed to nothing retires the own segment, no loan.
    pool.tick.store(2, Ordering::Relaxed);
    assert_eq!(pool.move_to(0, &carriers, 1).retired, 3, "each bank's own segment");
    pool.housekeep(2, true);
    pool.banks().for_each(|(_, s)| assert_eq!(borrowed(s), vec![4, 5]));
    // A chunk reaching the second level's first loan: that level goes.
    let chunk = mapped - 4 * segment + PAGE;
    assert_eq!(pool.take_back(&region, chunk, 3, u64::MAX).expect("taken back"), 3);
    assert_eq!(region.bound(), chunk);
    for (_, s) in pool.banks() {
        assert_eq!(borrowed(s), vec![5], "the first level stays lent");
        assert!((0..3).all(|e| s.resident_slot((0, e)).is_some_and(|slot| slot >= 15)), "and keeps its experts");
        assert!((0..3).all(|e| !s.is_resident((1, e))), "the level handed back lost its own");
    }
    region.object(chunk, Use::Scratch, "chunk").expect("the chunk's region is whole");
    assert_eq!(pool.lend(&region, floor).expect("and lent again"), 3);
    assert_eq!(pool.take_back(&region, mapped, 4, u64::MAX).expect("all of it"), 6);
    assert_eq!((region.bound(), region.levels().lent), (mapped, 0));
    let _ = std::fs::remove_file(&path);
}
