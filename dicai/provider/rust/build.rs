use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?)
        .join("../../proto");

    let proto_file = proto_dir.join("layer.proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&[proto_file], &[proto_dir])?;

    Ok(())
}
