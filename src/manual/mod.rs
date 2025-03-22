mod barnes_hut_seq;
mod barnes_hut_pool;
mod brute_force;

pub use brute_force::*;

pub use barnes_hut_seq::*;

#[cfg(not(target_arch = "wasm32"))]
pub use barnes_hut_pool::*;
