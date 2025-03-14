use std::{path::PathBuf, process::Command, time::SystemTime};

const SLANG_VERSION: &str = "2025.5.3";

fn main() {
    let out_dir: PathBuf = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let slang_dir: PathBuf = out_dir.join("slang");
    let slang_tarball: PathBuf = out_dir.join("slang.tar.gz");
    let slangc_binary_path: PathBuf = slang_dir.join("bin/slangc");

    let shader_dir = PathBuf::from("src/shaders");
    let shader_modules_dir = shader_dir.join("modules");
    let shader_output_dir = PathBuf::from("src/out");

    println!();

    (|| {
        if slangc_binary_path.exists() {
            let output = Command::new(&slangc_binary_path)
                .arg("-v")
                .output()
                .expect("Failed to execute command");
            let slangc_version = std::str::from_utf8(&output.stderr)
                .expect("Failed to convert slangc version to str");
            if slangc_version == SLANG_VERSION {
                println!(
                    "slangc binary is up date with current version {}!",
                    SLANG_VERSION
                );
                return;
            }
            println!(
                "slangc binary is not up to date with outdated version {}!",
                slangc_version
            );
        } else {
            println!("slangc binary does not exist!");
        }

        println!("Downloading slangc binary...");

        if slang_dir.exists() {
            println!("Removing old slang directory...");
            std::fs::remove_dir_all(&slang_dir).expect("Failed to remove slang directory");
        }
        std::fs::create_dir(&slang_dir).expect("Failed to create slang directory");

        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;

        if ["linux", "macos", "windows"].iter().all(|&x| os != x) {
            panic!("Unsupported OS");
        }

        if ["x86_64", "aarch64"].iter().all(|&x| arch != x) {
            panic!("Unsupported architecture");
        }

        let slang_url = format!(
            "https://github.com/shader-slang/slang/releases/download/v{0}/slang-{0}-{1}-{2}.tar.gz",
            SLANG_VERSION, os, arch
        );

        println!("Downloading slang binary from {}", slang_url);

        // Download the slang binary with curl to the .buildtools directory
        if !Command::new("curl")
            .arg("-L")
            .arg(&slang_url)
            .arg("-o")
            .arg(&slang_tarball)
            .status()
            .expect("Failed to execute command")
            .success()
        {
            panic!("Failed to download {}", slang_url);
        }

        // Extract the slang files to the slang directory
        if !Command::new("tar")
            .arg("-xvf")
            .arg(&slang_tarball)
            .arg("-C")
            .arg(slang_dir.display().to_string())
            .status()
            .expect("Failed to execute command")
            .success()
        {
            panic!("Failed to extract {:?}", slang_tarball.file_name().unwrap());
        }

        std::fs::remove_file(&slang_tarball).expect("Failed to remove slang tarball");

        println!("Installed slangc binary to {:?}!", slangc_binary_path);
    })();

    println!();
    println!("cargo::rerun-if-changed={}", shader_dir.display());
    println!();

    // Get list of shader files in the shader directory that end in .slang
    let shader_files = shader_dir
        .read_dir()
        .expect("Failed to read shader directory")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension() == Some(std::ffi::OsStr::new("slang")))
        .collect::<Vec<PathBuf>>();

    // Used to check if any shader modules are more recent than any output files
    let mut output_latest_modified = SystemTime::UNIX_EPOCH;

    let shader_to_output = shader_files.iter().map(|shader_file| {
        (
            shader_output_dir.join(format!(
                "{}.wgsl",
                shader_file.file_stem().unwrap().to_str().unwrap()
            )),
            shader_file,
        )
    });

    let needs_recompile = shader_to_output
        .filter(|(output_file, shader_file)| {
            if !output_file.exists() {
                return true;
            }
            let shader_metadata = shader_file.metadata().unwrap();
            let output_metadata = output_file.metadata().unwrap();
            if let Ok(shader_modified) = shader_metadata.modified() {
                if let Ok(output_modified) = output_metadata.modified() {
                    output_latest_modified = output_latest_modified.max(output_modified);
                    if output_modified < shader_modified {
                        return true;
                    }
                }
            }
            println!(
                "Skipping shader {:?} because {:?} is up to date",
                shader_file, output_file
            );
            false
        })
        .collect::<Vec<_>>();

    let modules_modified = match shader_modules_dir.read_dir() {
        Ok(modules) => modules
            .map(|entry| entry.unwrap().metadata().unwrap().modified().unwrap())
            .max()
            .unwrap(),
        Err(_) => SystemTime::UNIX_EPOCH,
    };

    if output_latest_modified != SystemTime::UNIX_EPOCH && modules_modified > output_latest_modified
    {
        println!("\nNevermind! Recompiling all shaders because modules have been modified\n");
        // We need to delete the output directory because changes in modules could be superficial
        std::fs::remove_dir_all(&shader_output_dir)
            .expect("Failed to remove shader output directory");
    }

    if !shader_output_dir.exists() {
        std::fs::create_dir(&shader_output_dir).expect("Failed to create shader output directory");
    }

    let slangc_processes = ({
        if modules_modified > output_latest_modified {
            // Recreate the list because it was consumed by the filter
            shader_files
                .iter()
                .map(|shader_file| {
                    (
                        shader_output_dir.join(format!(
                            "{}.wgsl",
                            shader_file.file_stem().unwrap().to_str().unwrap()
                        )),
                        shader_file,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            needs_recompile
        }
    })
    .into_iter()
    .map(|(output_file, shader_file)| {
        println!("Compiling shader {:?} to {:?}", shader_file, output_file);
        let child = Command::new(&slangc_binary_path)
            .arg(shader_file)
            .arg("-target")
            .arg("wgsl")
            .arg("-o")
            .arg(&output_file)
            .spawn()
            .expect("Failed to execute command");
        (shader_file, output_file, child)
    })
    .collect::<Vec<_>>();

    println!();

    for (shader_file, output_file, mut process) in slangc_processes {
        if !process
            .wait()
            .expect("Failed to wait for slangc process")
            .success()
        {
            panic!(
                "Failed to compile shader {:?} to {:?}",
                shader_file, output_file
            );
        }
        println!("Compiled shader {:?} to {:?}", shader_file, output_file);
    }
}
