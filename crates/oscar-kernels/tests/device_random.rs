//! A binary of its own, as a process's tests share the device's one seed.
use candle_core::Tensor;

#[test]
fn a_fixed_seed_draws_the_same_row() {
    let device = oscar_offload::pools::test_device().expect("Metal device required");
    let draw = || {
        device.set_seed(3).unwrap();
        Tensor::randn(0f32, 4f32, 248_320, &device).unwrap().to_vec1::<f32>().unwrap()
    };
    let (a, b) = (draw(), draw());
    let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    assert_eq!(differ, 0, "{differ} of {} positions differ between two draws at one seed", a.len());
}
