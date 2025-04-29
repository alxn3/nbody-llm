# N-Body Simulation

This repository contains code that does N-body simulation on a variable number of points. It is able to be configured to use different methods such as barnes-hut or brute-force each with adjustable parameters if applicable. There are two different implementations of the barnes-hut algorithm--one created manually and another through strictly LLMs with minimal human intervention.

## Running

To run the project, just type the following command in the terminal:
```sh
cargo run
```

By default, the project runs with the visualizer. To disable all visualizations, please add the `--no-default-features` flag when running.

### Web

Requires [Trunk](https://github.com/trunk-rs/trunk) to be installed.

The following command will start a server at `localhost:8080`:
```sh
trunk serve
```

### WSL

Specifically for the current Ubuntu 24.04 distro for WSL, there is an issue with wayland so it needs to be disabled to run correctly. To do this, unset the `WAYLAND_DISPLAY` environmental variable.

```sh
WAYLAND_DISPLAY= cargo run
```
