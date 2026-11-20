//! The engine: the offload dispatcher the load installs for every MoE forward, its event lane and its helpers.

pub mod approx;
pub mod dispatcher;
pub mod eval;
pub mod event_lane;
pub(crate) mod router;

/// The load's, once stage one built it.
pub(crate) static PAGING_DISPATCHER: std::sync::OnceLock<std::sync::Arc<dispatcher::OffloadDispatcher>> =
    std::sync::OnceLock::new();

pub fn offload_dispatcher() -> Option<&'static std::sync::Arc<dispatcher::OffloadDispatcher>> {
    PAGING_DISPATCHER.get()
}

/// The GPU waits on the resolver every layer, and runs a command buffer at the QoS of the thread that commits it.
pub(crate) fn set_interactive_qos() {
    // <pthread/qos.h>: QOS_CLASS_USER_INTERACTIVE = 0x21.
    extern "C" {
        fn pthread_set_qos_class_self_np(
            qos_class: std::ffi::c_uint, relative_priority: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
    // Best-effort; a failure leaves the default QoS.
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x21, 0);
    }
}
