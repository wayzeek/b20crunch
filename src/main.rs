use b20crunch::{b20, mine, words};
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
    #[arg(long)]
    workers: Option<usize>,
    /// Output JSONL file
    #[arg(long, default_value = "hits.jsonl")]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().cmd {
        Cmd::Mine(a) => {
            let deployer = b20::parse_address(&a.deployer).map_err(anyhow::Error::msg)?;
            let words = words::parse_words(&a.words).map_err(anyhow::Error::msg)?;
            mine::run(mine::MineOpts {
                deployer,
                words,
                positions: a.positions,
                inner_min: a.inner_min,
                start: a.start,
                count: a.count,
                workers: a.workers.unwrap_or_else(num_cpus::get),
                out: a.out,
            })?;
            Ok(())
        }
    }
}
