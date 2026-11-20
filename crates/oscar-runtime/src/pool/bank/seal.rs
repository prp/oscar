//! The seal: a layer's experts pinned from the low end, each from the cheapest place (where it sits, a copy, a read).

use super::*;

/// A seal's reader threads, each with a bounce buffer the host line charges.
pub const SEAL_READERS: usize = 8;

impl Bank {
    /// Returns the experts copied and read; counts no miss or fetch.
    pub fn seal(&self, keys: &[LayerExpertKey], current_tick: u32) -> Result<(usize, usize)> {
        let refuse = || Error::Offload(oscar_offload::Error::Layout("seal: no keys".into()));
        let layer = keys.first().map(|k| k.0).ok_or_else(refuse)?;
        let extents = keys.iter().map(|&k| self.extent(k)).collect::<Result<Vec<_>>>()?;
        // (slot, key, extent, the slot copied from); the layer's own slots, pinned where they sit.
        let mut jobs: Vec<(usize, LayerExpertKey, ExpertExtent, Option<usize>)> = Vec::with_capacity(keys.len());
        let mut kept: Vec<usize> = Vec::new();
        let mut st = self.lock();
        while keys.iter().any(|&k| self.landing(k)) {
            st = self.cv.wait(st).expect("HotPoolBank state lock poisoned");
        }
        let deadline = Instant::now() + self.stall_cap();
        // No slot the device has not passed: its fence's command buffer may wait on this seal's own keys.
        let (own, free) = loop {
            let (mut own, mut free) = (HashMap::<u16, usize>::new(), Vec::<usize>::new());
            let (reached, per) = (self.reached(), st.per_segment);
            let mut s = 0;
            while s < self.index && free.len() + own.len() < keys.len() {
                let seg = &st.segs[s / per];
                if !seg.standing() || seg.borrowed {
                    s = (s / per + 1) * per;
                    continue;
                }
                if self.slots[s].pins.load(Ordering::Acquire) == 0 {
                    // Own only where published: a duplicate is takeable like any other.
                    match st.owners[s].key {
                        Some(o) if self.landing(o) => {}
                        Some(o) if o.0 == layer && self.word_of(o) == s as i32 => {
                            own.insert(o.1, s);
                        }
                        _ if self.passed(s, reached) => free.push(s),
                        _ => {}
                    }
                }
                s += 1;
            }
            if free.len() >= keys.iter().filter(|k| !own.contains_key(&k.1)).count() {
                break (own, free);
            }
            if Instant::now() >= deadline {
                return Err(Error::Exhausted { hot: self.capacity() });
            }
            st = self.stall(st, current_tick);
        };
        let mut free = free.into_iter();
        for (&key, &extent) in keys.iter().zip(&extents) {
            let per = st.per_segment;
            if let Some(&slot) = own.get(&key.1) {
                self.slots[slot].pins.fetch_add(1, Ordering::AcqRel);
                st.segs[slot / per].sealed += 1;
                kept.push(slot);
                continue;
            }
            let slot = free.next().expect("the takeable slots were counted");
            if st.owners[slot].key.is_some() {
                self.evict(&mut st, slot, current_tick);
            } else {
                self.slots[slot].evicted_layer.store(u16::MAX, Ordering::Release);
            }
            // Resident elsewhere: copied, its source pinned through the copy.
            let from = self.resident_slot(key).filter(|&s| s != slot && self.present(&st, s));
            if let Some(from) = from {
                self.slots[from].pins.fetch_add(1, Ordering::AcqRel);
            }
            self.claim(&mut st, slot, key, current_tick, u32::MAX);
            self.slots[slot].pins.fetch_add(1, Ordering::AcqRel);
            if from.is_none() {
                self.mark(key, filling(slot));
            }
            st.segs[slot / per].sealed += 1;
            jobs.push((slot, key, extent, from));
        }
        drop(st);
        // A few readers, as the drive is the bound; each slot written once the device and the drafter are past it.
        let chunk = jobs.len().div_ceil(SEAL_READERS.min(jobs.len().max(1))).max(1);
        let read = std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = jobs
                .chunks(chunk)
                .map(|part| {
                    s.spawn(move || -> Result<()> {
                        for &(slot, _, extent, from) in part {
                            self.wait_gpu_safe(slot);
                            self.wait_draft_safe(slot);
                            let Some(from) = from else {
                                self.read_into(0, slot, extent)?;
                                continue;
                            };
                            // SAFETY: one reader per slot the seal holds; the source is pinned and only read; the bank
                            // refuses an extent longer than a slot.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    self.segments.ptr(from),
                                    self.segments.ptr(slot),
                                    extent.byte_count,
                                )
                            };
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                let reader = h.join().map_err(|_| oscar_offload::Error::Layout("seal reader panicked".into()))?;
                reader?;
            }
            Ok(())
        });
        let mut st = self.lock();
        let unpin = |slot: usize| self.slots[slot].pins.fetch_sub(1, Ordering::AcqRel);
        // A failed read undoes the reservation: every slot unpinned, each read one freed with its fence standing.
        if let Err(e) = read {
            let per = st.per_segment;
            for slot in kept.iter().copied().chain(jobs.iter().map(|j| j.0)) {
                unpin(slot);
                st.segs[slot / per].sealed -= 1;
            }
            for &(slot, key, _, from) in &jobs {
                match from {
                    Some(from) => {
                        unpin(from);
                    }
                    None => self.mark(key, ABSENT),
                }
                st.assign(slot, None);
            }
            drop(st);
            self.cv.notify_all();
            return Err(e);
        }
        for &(slot, key, _, from) in &jobs {
            self.slots[slot].stamp.store(stamp_of(key), Ordering::Release);
            self.mark(key, slot as i32);
            if let Some(from) = from {
                unpin(from);
            }
        }
        drop(st);
        self.cv.notify_all();
        let copied = jobs.iter().filter(|j| j.3.is_some()).count();
        Ok((copied, jobs.len() - copied))
    }
}
