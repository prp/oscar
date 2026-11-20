//! The four refusals end to end and then the emergency un-wire, alone in a process: every bound here is process-wide.

use candle_metal_kernels::metal::{BufferPool, Commands, Device, Fences, Use};
use oscar_offload::fixed::fixed_pool;
use oscar_offload::machine;
use oscar_offload::segments::{Addressing, SegmentPool, PAGE};
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

#[test]
fn every_bound_refuses_a_claim_rather_than_compensating_for_it() {
    let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("a Metal device");
    the_plans_region_refuses_a_cut_it_cannot_hold_and_maps_nothing_for_it();
    every_bound_refuses_the_segment_rather_than_taking_the_machine(&device);
    the_emergency_gives_up_what_this_process_holds(&device);
}

/// Never compensated by mapping more or by waiting: room comes from objects dropping.
fn the_plans_region_refuses_a_cut_it_cannot_hold_and_maps_nothing_for_it() {
    let dev = Device::system_default().expect("a Metal device");
    let queue = dev.new_command_queue().expect("a queue");
    let _commands = Commands::new(queue.clone(), 0, None).expect("commands");
    // Progress moved by hand: nothing is encoded, so the release fence is the committed count.
    let completed = objc2_metal::MTLDevice::newSharedEvent(dev.as_ref()).expect("a shared event");
    let fences =
        Fences::new(Arc::new(AtomicU64::new(0)), completed.clone(), 0, Arc::new(vec![Arc::new(AtomicUsize::new(0))]));
    let pools = oscar_offload::pools::install();
    pools.attach(&queue, 0);

    // Four objects of eight pages and not a page more.
    pools.plan_region(32 * PAGE).expect("the region");
    let held: Vec<_> = (0..4)
        .map(|_| pools.allocate(&dev, 8 * PAGE, Use::Scratch, 0, &fences, "test").expect("inside the region"))
        .collect();
    let mapped = machine::PLAN.held();
    let why = pools
        .allocate(&dev, 8 * PAGE, Use::Scratch, 0, &fences, "test")
        .expect_err("a fifth object has no run")
        .to_string();
    assert!(why.contains("working region"), "the refusal must name the region: {why}");
    assert_eq!(machine::refusals().0, [0, 1, 0, 0], "the region refused it, not the process's line");
    assert_eq!(machine::PLAN.held(), mapped, "a refusal maps nothing");

    // Dropped runs are cut again at once at any size: a run is pages, not a class.
    drop(held);
    let again = pools.allocate(&dev, 16 * PAGE, Use::Scratch, 0, &fences, "test").expect("a re-cut at another size");
    assert_eq!(machine::PLAN.held(), mapped, "nothing new was mapped for it");
    drop(again);

    // A cut wider than any free run is refused, however much is free elsewhere.
    let _half = pools.allocate(&dev, 20 * PAGE, Use::Scratch, 0, &fences, "test").expect("twenty of thirty-two");
    let why = pools
        .allocate(&dev, 16 * PAGE, Use::Scratch, 0, &fences, "test")
        .expect_err("sixteen pages with twelve free")
        .to_string();
    assert!(why.contains("working region"), "{why}");
    assert_eq!(machine::refusals().0, [0, 2, 0, 0]);
    assert_eq!(machine::PLAN.held(), mapped, "and still nothing was mapped");
}

fn every_bound_refuses_the_segment_rather_than_taking_the_machine(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) {
    let pool = fixed_pool(device).expect("the fixed pool");

    // The plan's ceiling, the bound that should decide, at what is held already.
    machine::PLAN.arm(machine::PLAN.held().max(1));
    let why = pool.take(device, 64 << 20).expect_err("a claim past the ceiling is refused").to_string();
    assert!(why.contains("exceeds plan ceiling"), "the refusal must say the plan refused: {why}");
    machine::PLAN.arm(0);
    // Counted, not inferred: an uncounted refusal looks like a branch that never fired.
    assert_eq!(machine::refusals().0, [1, 2, 0, 0], "the ceiling's refusal is the ceiling's");

    // The machine's floor, the backstop, at every byte the machine has.
    machine::set_floor(u64::MAX / 2);
    let why = pool.take(device, 64 << 20).expect_err("a segment past the floor is refused").to_string();
    assert!(why.contains("floor"), "the refusal must say the machine refused: {why}");
    machine::set_floor(0);
    assert_eq!(machine::refusals().0, [1, 2, 1, 0], "the floor's refusal is the floor's");

    // Neither armed, the same request is served.
    let held = machine::PLAN.held();
    let (_buffer, bytes) = pool.take(device, 64 << 20).expect("the same segment, both bounds down");
    assert_eq!(bytes, 64 << 20);
    assert_eq!(machine::PLAN.held(), held + bytes as u64, "a served take holds what it mapped");

    // The kernel's lock, which a solved plan never reaches, at one page: the segment stays absent, not present unwired.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let two = SegmentPool::new(device, page, 2, 2, Addressing::Table).expect("two segments of two pages");
    let mut was: libc::rlimit = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut was) }, 0);
    let one_page = libc::rlimit { rlim_cur: page as libc::rlim_t, rlim_max: was.rlim_max };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &one_page) }, 0);
    let held = machine::PLAN.held();
    let why = two.take(device, 0, 2, true).expect_err("a lock the kernel refuses fails the take").to_string();
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &was) }, 0);
    assert!(why.contains("wire limit"), "the refusal must say the allocation left the plan: {why}");
    assert_eq!(two.live_units(), 0, "the refused take left no units behind");
    assert_eq!(machine::PLAN.held(), held, "the refused take gave its claim back");
    assert_eq!(machine::refusals().0, [1, 2, 1, 1], "the kernel's refusal is the lock's");

    two.take(device, 0, 2, true).expect("the same segment, the limit lifted");
    two.release(0).expect("released");

    // A table pool's take, as the expert pool's banks make them, through the same line.
    let table = SegmentPool::new(device, PAGE, 2, 4, Addressing::Table).expect("two segments of four units");
    machine::PLAN.arm(machine::PLAN.held().max(1));
    let held = machine::PLAN.held();
    let why = table.take(device, 0, 4, true).expect_err("a segment past the ceiling is refused").to_string();
    machine::PLAN.arm(0);
    assert!(why.contains("exceeds plan ceiling"), "the refusal must say the plan refused: {why}");
    assert_eq!(machine::refusals().0, [2, 2, 1, 1], "the ceiling's refusal is the ceiling's");
    assert_eq!((table.live_units(), machine::PLAN.held()), (0, held), "the refused take left nothing behind");
    table.take(device, 0, 4, true).expect("the same segment, the line down");
    assert_eq!(table.live_units(), 4);
    assert_eq!(machine::PLAN.held(), held + 4 * PAGE as u64, "a served take holds what it mapped");
}

/// Last, as it gives up the process's pools; its evidence is the kernel's count of this process's locks.
fn the_emergency_gives_up_what_this_process_holds(device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>) {
    let pool = fixed_pool(device).expect("the fixed pool");
    let (_buffer, bytes) = pool.take(device, 64 << 20).expect("a segment to give up");
    let held = machine::PLAN.held();
    let given_up = oscar_offload::segments::release_all();
    assert_eq!(given_up.missed, 0, "every pool was reachable on a quiet machine");
    let fell = given_up.locked.0 - given_up.locked.1;
    assert!(fell >= bytes as u64, "the kernel's count of this process's locks fell by under the segment: {given_up}");
    // The mappings survive it, still counted: the address space dies with the process.
    assert_eq!(machine::PLAN.held(), held, "the pages are still mapped and still counted");
}
