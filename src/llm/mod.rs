mod barnes_hut;
mod barnes_hut_midterm;

// Export only the specific types we need from barnes_hut
pub use barnes_hut::{create_barnes_hut_3d, BarnesHut3D, BarnesHutSimulation};

// Export with a different name to avoid conflicts
pub use barnes_hut_midterm::BarnesHutSimulationMidterm;
