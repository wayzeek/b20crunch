use b20crunch::{b20, chain, mine, words};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "b20crunch",
    version,
    about = "Finds salts that spell words in B20 token addresses on Base"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Mine salts whose B20 address spells one of your words
    Mine(MineArgs),
    /// Check a salt against the live factory (read-only)
    Verify(VerifyArgs),
    /// Pre-flight (default) or broadcast a B20 ASSET deployment
    Deploy(DeployArgs),
}

#[derive(Args)]
struct VerifyArgs {
    /// EOA that will call createB20 directly
    #[arg(long)]
    deployer: String,
    /// Salt as a decimal integer
    #[arg(long)]
    salt: u128,
    /// Expected ASSET address; exit nonzero on mismatch
    #[arg(long)]
    expect: Option<String>,
    #[arg(long, default_value = b20crunch::chain::DEFAULT_RPC)]
    rpc: String,
}

#[derive(Args)]
struct DeployArgs {
    /// EOA that will call createB20 directly
    #[arg(long)]
    deployer: String,
    /// Salt as a decimal integer
    #[arg(long)]
    salt: u128,
    /// Expected ASSET address; every pre-flight step must agree
    #[arg(long)]
    expect: String,
    /// Token name
    #[arg(long)]
    name: String,
    /// Token symbol
    #[arg(long)]
    symbol: String,
    #[arg(long, default_value_t = 18)]
    decimals: u8,
    #[arg(long, default_value = b20crunch::chain::DEFAULT_RPC)]
    rpc: String,
    /// Actually broadcast (needs B20_DEPLOYER_KEY); without it, dry-run only
    #[arg(long)]
    send: bool,
}

#[derive(Args)]
struct MineArgs {
    /// EOA that will call createB20 directly (multisig/proxy voids the salt)
    #[arg(long)]
    deployer: String,
    /// Comma-separated hex words (0-9 a-f; leetspeak: o=0 l/i=1 s=5 t=7 g=6 z=2)
    #[arg(long)]
    words: String,
    /// Where the word must sit in the 18-char window
    #[arg(long, value_enum, default_value_t = words::Positions::Ends)]
    positions: words::Positions,
    /// Minimum word length matched mid-window (with --positions any)
    #[arg(long, default_value_t = 6)]
    inner_min: usize,
    /// First salt to scan (resume offset)
    #[arg(long, default_value_t = 0)]
    start: u128,
    /// Total salts to scan across all workers (default: run until Ctrl-C)
    #[arg(long)]
    count: Option<u64>,
    /// Worker threads (default: logical cores)
    #[arg(long, conflicts_with = "gpu")]
    workers: Option<usize>,
    /// Mine on a GPU via OpenCL instead of CPU threads (build with --features gpu)
    #[arg(long)]
    gpu: bool,
    /// GPU device index when several are present (with --gpu)
    #[arg(long, requires = "gpu")]
    device: Option<usize>,
    /// Salts per GPU dispatch, for tuning (with --gpu)
    #[arg(long, requires = "gpu")]
    gpu_batch: Option<u64>,
    /// Output JSONL file
    #[arg(long, default_value = "hits.jsonl")]
    out: PathBuf,
    /// After the run, cross-check every hit against the live factory
    #[arg(long)]
    verify: bool,
    /// RPC endpoint, used only with --verify
    #[arg(long, default_value = b20crunch::chain::DEFAULT_RPC)]
    rpc: String,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Mine(a) => {
            let deployer = b20::parse_address(&a.deployer).map_err(anyhow::Error::msg)?;
            let words = words::parse_words(&a.words).map_err(anyhow::Error::msg)?;
            let backend = if a.gpu {
                mine::Backend::Gpu(mine::GpuConfig {
                    device: a.device,
                    batch: a.gpu_batch,
                    ..Default::default()
                })
            } else {
                mine::Backend::Cpu {
                    workers: a.workers.unwrap_or_else(num_cpus::get),
                }
            };
            let hits = mine::run(mine::MineOpts {
                deployer,
                words,
                positions: a.positions,
                inner_min: a.inner_min,
                start: a.start,
                count: a.count,
                backend,
                out: a.out,
            })?;
            if a.verify {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(chain::verify_hits(&a.rpc, deployer, &hits))?;
            }
            Ok(())
        }
        Cmd::Verify(a) => {
            let deployer = b20::parse_address(&a.deployer).map_err(anyhow::Error::msg)?;
            let expect = a
                .expect
                .map(|e| b20::parse_address(&e).map_err(anyhow::Error::msg))
                .transpose()?;
            let rt = tokio::runtime::Runtime::new()?;
            let ok = rt.block_on(chain::verify(&a.rpc, deployer, a.salt, expect))?;
            if !ok {
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::Deploy(a) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(chain::deploy(chain::DeployOpts {
                rpc: a.rpc,
                deployer: b20::parse_address(&a.deployer).map_err(anyhow::Error::msg)?,
                salt: a.salt,
                expect: b20::parse_address(&a.expect).map_err(anyhow::Error::msg)?,
                name: a.name,
                symbol: a.symbol,
                decimals: a.decimals,
                send: a.send,
            }))
        }
    }
}
