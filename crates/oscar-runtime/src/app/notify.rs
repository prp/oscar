//! A new trouble line's macOS notification through UNUserNotificationCenter (its classes sent their messages, the
//! framework linked): only from a bundle, as the centre needs a bundle identifier; the leave asked by the system.

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send};
use objc2_foundation::{NSBundle, NSError, NSString};

#[link(name = "UserNotifications", kind = "framework")]
extern "C" {}

fn centre() -> Option<Retained<AnyObject>> {
    NSBundle::mainBundle().bundleIdentifier()?;
    Some(unsafe { msg_send![class!(UNUserNotificationCenter), currentNotificationCenter] })
}

/// At launch: alerts and sound (UNAuthorizationOptions 4 | 2); the system asks the user once.
pub fn request_leave() {
    let Some(centre) = centre() else { return };
    let done = RcBlock::new(|_granted: Bool, _error: *mut NSError| {});
    let _: () = unsafe { msg_send![&*centre, requestAuthorizationWithOptions: 6usize, completionHandler: &*done] };
}

pub fn post(text: &str) {
    let Some(centre) = centre() else { return };
    let content: Retained<AnyObject> = unsafe { msg_send![class!(UNMutableNotificationContent), new] };
    let _: () = unsafe { msg_send![&*content, setTitle: &*NSString::from_str("Oscar")] };
    let _: () = unsafe { msg_send![&*content, setBody: &*NSString::from_str(text)] };
    let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    let id = NSString::from_str(&format!("oscar-{at}"));
    let request: Retained<AnyObject> = unsafe {
        msg_send![class!(UNNotificationRequest), requestWithIdentifier: &*id, content: &*content, trigger: None::<&AnyObject>]
    };
    let done = RcBlock::new(|_error: *mut NSError| {});
    let _: () = unsafe { msg_send![&*centre, addNotificationRequest: &*request, withCompletionHandler: &*done] };
}
