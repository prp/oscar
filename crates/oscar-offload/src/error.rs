use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("metal error: {0}")]
    Metal(String),

    #[error("layout error: {0}")]
    Layout(String),

    /// A bound the plan drew turned a mapping down: a plan defect, not a device failure.
    #[error("segment pool: {0}")]
    Refused(crate::machine::Refusal),

    #[error("short read: expected {expected} bytes at offset {offset}, got {got}")]
    ShortRead { expected: usize, got: isize, offset: u64 },
}

/// By hand: thiserror's `#[from]` would make the refusal the error's source, which a chain prints twice.
impl From<crate::machine::Refusal> for Error {
    fn from(refusal: crate::machine::Refusal) -> Self {
        Self::Refused(refusal)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
