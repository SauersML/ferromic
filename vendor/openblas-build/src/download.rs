use anyhow::Result;
use std::path::{Path, PathBuf};
use ureq::{
    config::Config,
    tls::{RootCerts, TlsConfig, TlsProvider},
};

const OPENBLAS_VERSION: &str = "0.3.30";

pub fn openblas_source_url() -> String {
    format!(
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v{}/OpenBLAS-{}.tar.gz",
        OPENBLAS_VERSION, OPENBLAS_VERSION
    )
}

pub fn download(out_dir: &Path) -> Result<PathBuf> {
    let dest = out_dir.join(format!("OpenBLAS-{}", OPENBLAS_VERSION));
    if !dest.exists() {
        if let Err(err) = try_download(out_dir, RootCerts::PlatformVerifier) {
            eprintln!(
                "openblas-build: platform TLS verifier failed ({}), retrying with Mozilla roots",
                err
            );
            try_download(out_dir, RootCerts::WebPki)?;
        }
        assert!(dest.exists());
    }
    Ok(dest)
}

fn try_download(out_dir: &Path, roots: RootCerts) -> Result<()> {
    let buf = get_agent(roots)
        .get(&openblas_source_url())
        .call()?
        .into_body()
        .into_reader();
    let gz_stream = flate2::read::GzDecoder::new(buf);
    let mut ar = tar::Archive::new(gz_stream);
    ar.unpack(out_dir)?;
    Ok(())
}

fn get_agent(root_certs: RootCerts) -> ureq::Agent {
    Config::builder()
        .tls_config(
            TlsConfig::builder()
                .provider(TlsProvider::Rustls)
                .root_certs(root_certs)
                .build(),
        )
        .build()
        .new_agent()
}
