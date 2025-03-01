# N-Body Simulation

## Running

To run the project, just type the following command in the terminal:
```sh
cargo run
```

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
