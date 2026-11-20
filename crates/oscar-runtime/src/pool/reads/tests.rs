use super::*;
use crate::pool::bank::tests::{bank_of, test_lock, wait_until};
use crate::pool::bank::Plan;
use std::time::Duration;

/// A read the queue only orders, never makes.
fn req(bank: &Arc<Bank>, key_layer: u16, class: FetchClass, deadline: u32) -> FetchReq {
    let extent = ExpertExtent { shard_id: 0, file_offset: 0, byte_count: 2 * 1024 * 1024 };
    FetchReq::new(bank, (key_layer, 0), extent, key_layer as usize, class, deadline)
}

/// A read of `key` into the slot the demand plan reserved.
fn reserved(bank: &Arc<Bank>, key: LayerExpertKey, class: FetchClass) -> FetchReq {
    let Plan::Reserved { slot, extent } = bank.plan(&[key], 0).expect("a demand plan")[0] else {
        panic!("{key:?} is not cold");
    };
    FetchReq::new(bank, key, extent, slot, class, u32::MAX)
}

/// Each row's pushes against the keys it pops, in order; the length follows every push and pop. A blocked pop takes
/// the next push, and a shutdown wakes it empty.
#[test]
fn the_queue_pops_by_deadline_then_class_then_push_order() {
    let _g = test_lock();
    let b = bank_of(1, 1, [0]);
    use FetchClass::{Demand as D, Reserved as R};
    let n = u32::MAX;
    let rows: [(&str, &[(u16, FetchClass, u32)], &[u16]); 4] = [
        ("by deadline, demand due now first", &[(9, R, 9), (5, R, 5), (7, R, 7), (99, D, 0)], &[99, 5, 7, 9]),
        ("demand wins a deadline tie", &[(1, R, 5), (2, D, 5)], &[2, 1]),
        ("the deadline outranks the class", &[(4, D, 9), (3, R, 1)], &[3, 4]),
        (
            "an equal deadline pops in push order",
            &[(0, R, n), (1, R, n), (2, R, n), (3, R, n), (4, R, n)],
            &[0, 1, 2, 3, 4],
        ),
    ];
    for (row, pushes, pops) in rows {
        let q = FetchQueue::default();
        assert!(q.is_empty(), "{row}");
        q.push_many(pushes.iter().map(|&(key, class, deadline)| req(&b, key, class, deadline)));
        let mut got = Vec::new();
        while !q.is_empty() {
            assert_eq!(q.len(), pushes.len() - got.len(), "{row}");
            got.push(q.pop(false).expect("a pushed read").0.key.0);
        }
        assert!(q.pop(false).is_none(), "{row}");
        assert_eq!(got, pops, "{row}");
    }
    let q = Arc::new(FetchQueue::default());
    let popper = || {
        let q = Arc::clone(&q);
        std::thread::spawn(move || q.pop(true).map(|(r, _)| r.key.0))
    };
    let h = popper();
    std::thread::sleep(Duration::from_millis(10));
    q.push_many([req(&b, 42, FetchClass::Reserved, u32::MAX)]);
    assert_eq!(h.join().expect("the popper"), Some(42));
    let h = popper();
    std::thread::sleep(Duration::from_millis(20));
    q.shutdown();
    assert_eq!(h.join().expect("the popper"), None);
}

/// Each fill published and counted by its class, a failed read counted and aborted; the drop wakes the parked readers.
#[test]
fn the_readers_publish_each_fill_by_class_and_abort_a_failed_one() {
    let _g = test_lock();
    let none = std::panic::catch_unwind(|| Readers::new(Arc::new(FetchQueue::default()), &[0]));
    assert!(none.is_err(), "a pool of no reader is refused");
    let (bank, c0) = (bank_of(6, 6, 0..6), COUNTERS.now());
    let queue = Arc::new(FetchQueue::default());
    let readers = Readers::new(Arc::clone(&queue), &[2]);
    // A shard the file lacks: the read fails.
    let mut failing = reserved(&bank, (0, 5), FetchClass::Reserved);
    failing.shard_id = 1;
    queue.push_many([reserved(&bank, (0, 0), FetchClass::Demand), failing]);
    queue.push_many((1..5).map(|k| reserved(&bank, (0, k), FetchClass::Reserved)));
    let done = || (0..5).all(|k| bank.is_resident((0, k))) && COUNTERS.now().read_errors > c0.read_errors;
    assert!(wait_until(done), "the reads never all landed");
    let t0 = std::time::Instant::now();
    drop(readers);
    assert!(t0.elapsed() < Duration::from_secs(2), "the drop blocked on the parked readers");
    let c = COUNTERS.now();
    let by_class = (c.completed_demand, c.completed_prefetch, c.read_errors);
    assert_eq!(by_class, (c0.completed_demand + 1, c0.completed_prefetch + 4, c0.read_errors + 1));
    assert!(!bank.is_resident((0, 5)), "the failed read published nothing");
    assert!(matches!(bank.plan(&[(0, 5)], 0).unwrap()[0], Plan::Reserved { .. }), "the abort freed the reservation");
}
