//! Start at Login through SMAppService: the bundle registers itself as a login item; outside a bundle the system's
//! error says so.

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::NSError;

#[link(name = "ServiceManagement", kind = "framework")]
extern "C" {}

fn service() -> Retained<AnyObject> {
    unsafe { msg_send![class!(SMAppService), mainAppService] }
}

/// SMAppServiceStatusEnabled.
pub fn enabled() -> bool {
    let status: isize = unsafe { msg_send![&*service(), status] };
    status == 1
}

pub fn set(on: bool) -> anyhow::Result<()> {
    let service = service();
    let done: Result<(), Retained<NSError>> = unsafe {
        if on {
            msg_send![&*service, registerAndReturnError: _]
        } else {
            msg_send![&*service, unregisterAndReturnError: _]
        }
    };
    done.map_err(|e| anyhow::anyhow!("Start at Login: {}", e.localizedDescription()))
}
