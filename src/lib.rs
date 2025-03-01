pub mod render;
pub mod shared;

#[macro_export]
macro_rules! from_slang {
    ($($token:tt)*) => {
        concat!("out/",$($token)*,".wgsl")
    };
}
