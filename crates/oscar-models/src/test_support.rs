//! The crate's test fixtures.

use candle_core::{DType, Device, Result, Tensor};

use crate::state_slots::{Emit, RecurrentState};

/// Signed values up to 2.8, `seed` decorrelating one tensor from the next.
pub(crate) fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 7919 + seed * 104729) % 10007) as f32 - 5003.0) / 1777.0).collect()
}

pub(crate) fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.flatten_all()?.to_vec1()
}

pub(crate) fn assert_bitwise(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    let (got, want) = (host(got)?, host(want)?);
    assert_eq!(got.len(), want.len(), "{what}: lengths differ");
    let bad = got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
    assert_eq!(bad, 0, "{what}: {bad} of {} values differ", got.len());
    Ok(())
}

/// A first chunk's state: a zero conv window `(batch, channels, width)` over `recurrent`.
pub(crate) fn fresh_state(device: &Device, conv: (usize, usize, usize), recurrent: Tensor) -> Result<RecurrentState> {
    let conv_state = Tensor::zeros(conv, DType::F32, device)?;
    Ok(RecurrentState { conv_state, recurrent_state: recurrent, seqlen_offset: 0, emits: Vec::new() })
}

/// A prompt answers the same however it is cut: `x` `(b, s, c)` whole and at each of `cuts`, compared bitwise; the
/// whole run emits at every cut, each emitted conv window the state the cut there carries.
pub(crate) fn assert_cuts_match_whole(
    x: &Tensor, cuts: &[&[usize]], fresh: impl Fn() -> Result<RecurrentState>,
    conv: impl Fn(&Tensor, &mut RecurrentState) -> Result<Tensor>,
) -> Result<()> {
    let bits = |t: &Tensor| -> Result<Vec<u32>> { Ok(host(t)?.iter().map(|v| v.to_bits()).collect()) };
    let (_, s, c) = x.dims3()?;
    let mut whole = fresh()?;
    let mut ats: Vec<usize> = cuts.iter().flat_map(|c| c.iter().copied()).collect();
    ats.sort_unstable();
    ats.dedup();
    let emit =
        |at: usize| Ok(Emit { at, conv: whole.conv_state.zeros_like()?, recurrent: whole.recurrent_state.clone() });
    whole.emits = ats.into_iter().map(emit).collect::<Result<_>>()?;
    let want = bits(&conv(x, &mut whole)?)?;
    let want_state = bits(&whole.conv_state)?;
    for cuts in cuts {
        let (mut cache, mut at) = (fresh()?, 0usize);
        for &end in cuts.iter().chain(std::iter::once(&s)) {
            let n = end - at;
            // The chunk's (b, n, c) outputs against the whole's (b, s, c).
            for (i, g) in bits(&conv(&x.narrow(1, at, n)?, &mut cache)?)?.into_iter().enumerate() {
                let (bi, token, ch) = (i / (n * c), at + i / c % n, i % c);
                let at_whole = want[(bi * s + token) * c + ch];
                assert_eq!(g, at_whole, "{:?} cuts {cuts:?}: token {token} channel {ch}", x.device());
            }
            if let Some(e) = whole.emits.iter().find(|e| e.at == end) {
                assert_eq!(bits(&cache.conv_state)?, bits(&e.conv)?, "{:?}: the conv emitted at {end}", x.device());
            }
            cache.seqlen_offset += n;
            at = end;
        }
        assert_eq!(bits(&cache.conv_state)?, want_state, "{:?} cuts {cuts:?}: the carried state", x.device());
    }
    Ok(())
}
