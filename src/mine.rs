use crate::words::Positions;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pos {
    Prefix,
    Suffix,
    Inner,
}

impl Pos {
    pub fn as_str(&self) -> &'static str {
        match self {
            Pos::Prefix => "prefix",
            Pos::Suffix => "suffix",
            Pos::Inner => "inner",
        }
    }
}

enum Kind {
    /// All words share one length and inner matching is off: binary search per end.
    Uniform { len: usize, sorted: Vec<String> },
    /// Longest-first linear scans, then inner scan.
    General {
        words: Vec<String>,
        inner: Vec<String>,
    },
}

pub struct Matcher {
    kind: Kind,
    check_prefix: bool,
    check_suffix: bool,
}

impl Matcher {
    /// `words` must come from words::parse_words (lowercase, longest-first).
    pub fn new(words: &[String], positions: Positions, inner_min: usize) -> Matcher {
        let (check_prefix, check_suffix) = match positions {
            Positions::Prefix => (true, false),
            Positions::Suffix => (false, true),
            Positions::Ends | Positions::Any => (true, true),
        };
        let inner: Vec<String> = if positions == Positions::Any {
            words
                .iter()
                .filter(|w| w.len() >= inner_min)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let lens: std::collections::HashSet<usize> = words.iter().map(|w| w.len()).collect();
        let kind = if lens.len() == 1 && inner.is_empty() {
            let mut sorted = words.to_vec();
            sorted.sort();
            Kind::Uniform {
                len: *lens.iter().next().unwrap(),
                sorted,
            }
        } else {
            Kind::General {
                words: words.to_vec(),
                inner,
            }
        };
        Matcher {
            kind,
            check_prefix,
            check_suffix,
        }
    }

    /// At most one hit per tail: prefix, then suffix, then inner.
    pub fn find<'a>(&'a self, tail_hex: &[u8; 18]) -> Option<(Pos, &'a str)> {
        match &self.kind {
            Kind::Uniform { len, sorted } => {
                if self.check_prefix {
                    let head = &tail_hex[..*len];
                    if let Ok(i) = sorted.binary_search_by(|w| w.as_bytes().cmp(head)) {
                        return Some((Pos::Prefix, &sorted[i]));
                    }
                }
                if self.check_suffix {
                    let end = &tail_hex[18 - len..];
                    if let Ok(i) = sorted.binary_search_by(|w| w.as_bytes().cmp(end)) {
                        return Some((Pos::Suffix, &sorted[i]));
                    }
                }
                None
            }
            Kind::General { words, inner } => {
                if self.check_prefix {
                    for w in words {
                        if tail_hex.starts_with(w.as_bytes()) {
                            return Some((Pos::Prefix, w));
                        }
                    }
                }
                if self.check_suffix {
                    for w in words {
                        if tail_hex.ends_with(w.as_bytes()) {
                            return Some((Pos::Suffix, w));
                        }
                    }
                }
                for w in inner {
                    if tail_hex.windows(w.len()).any(|win| win == w.as_bytes()) {
                        return Some((Pos::Inner, w));
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::Positions;

    fn t(s: &str) -> [u8; 18] {
        s.as_bytes().try_into().unwrap()
    }

    #[test]
    fn uniform_fast_path_prefix_and_suffix() {
        let m = Matcher::new(&["dead".into(), "beef".into()], Positions::Ends, 6);
        assert_eq!(
            m.find(&t("dead12345678901234")),
            Some((Pos::Prefix, "dead"))
        );
        assert_eq!(
            m.find(&t("12345678901234beef")),
            Some((Pos::Suffix, "beef"))
        );
        assert_eq!(m.find(&t("123456dead89012345")), None); // inner not enabled
        assert_eq!(m.find(&t("123456789012345678")), None);
    }

    #[test]
    fn prefix_beats_suffix_beats_inner() {
        let m = Matcher::new(&["dead".into()], Positions::Ends, 4);
        // both ends match: prefix wins
        assert_eq!(
            m.find(&t("dead1234567890dead")),
            Some((Pos::Prefix, "dead"))
        );
        let m = Matcher::new(&["dead".into(), "c0ffee".into()], Positions::Any, 4);
        // suffix and inner both present: suffix wins
        assert_eq!(
            m.find(&t("12c0ffee901234dead")),
            Some((Pos::Suffix, "dead"))
        );
    }

    #[test]
    fn longest_word_wins_at_same_end() {
        let m = Matcher::new(&["deadbeef".into(), "dead".into()], Positions::Prefix, 6);
        assert_eq!(
            m.find(&t("deadbeef9012345678")),
            Some((Pos::Prefix, "deadbeef"))
        );
    }

    #[test]
    fn inner_respects_inner_min() {
        let m = Matcher::new(&["c0ffee".into(), "dead".into()], Positions::Any, 6);
        assert_eq!(
            m.find(&t("12c0ffee9012345678")),
            Some((Pos::Inner, "c0ffee"))
        );
        assert_eq!(m.find(&t("12dead345678901234")), None); // 4 < inner_min 6
    }

    #[test]
    fn prefix_only_and_suffix_only() {
        let m = Matcher::new(&["dead".into()], Positions::Prefix, 6);
        assert_eq!(m.find(&t("12345678901234dead")), None);
        let m = Matcher::new(&["dead".into()], Positions::Suffix, 6);
        assert_eq!(m.find(&t("dead12345678901234")), None);
    }
}

use crate::{b20, words};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use tiny_keccak::{Hasher, Keccak};

pub struct MineOpts {
    pub deployer: [u8; 20],
    pub words: Vec<String>,
    pub positions: words::Positions,
    pub inner_min: usize,
    pub start: u128,
    pub count: Option<u64>,
    pub workers: usize,
    pub out: PathBuf,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct HitRecord {
    pub word: String,
    pub position: String,
    pub salt: String,
    pub salt_bytes32: String,
    pub tail: String,
    pub asset_address: String,
    pub stablecoin_address: String,
}

struct RawHit {
    word: String,
    pos: Pos,
    salt: u128,
    tail: [u8; 9],
}

pub fn run(opts: MineOpts) -> anyhow::Result<Vec<HitRecord>> {
    anyhow::ensure!(opts.workers >= 1, "--workers must be at least 1");
    let matcher = Arc::new(Matcher::new(&opts.words, opts.positions, opts.inner_min));
    let stop = Arc::new(AtomicBool::new(false));
    let scanned = Arc::new(AtomicU64::new(0));
    // ctrlc handler may already be installed when run() is called twice in one
    // process (tests); ignore the AlreadyInstalled error.
    {
        let stop = stop.clone();
        let _ = ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed));
    }

    // combined expected work across words (approximate union = sum)
    let p_combined: f64 = opts
        .words
        .iter()
        .map(|w| 1.0 / words::expected_salts_per_hit(w.len(), opts.positions, opts.inner_min))
        .sum();
    eprintln!("word                placement  expected salts per hit");
    for w in &opts.words {
        let e = words::expected_salts_per_hit(w.len(), opts.positions, opts.inner_min);
        eprintln!(
            "{:<19} {:<10} ~{}",
            w,
            format!("{:?}", opts.positions).to_lowercase(),
            words::humanize(e)
        );
    }

    // exact split: total scanned == count; the first (count % workers) workers
    // take one extra salt so nothing overruns the requested total
    let quotas: Option<Vec<u64>> = opts.count.map(|c| {
        let base = c / opts.workers as u64;
        let extra = (c % opts.workers as u64) as usize;
        (0..opts.workers)
            .map(|w| base + u64::from(w < extra))
            .collect()
    });
    let (tx, rx) = mpsc::channel::<RawHit>();

    // writer thread: JSONL file + console line per hit
    let out_path = opts.out.clone();
    let writer = std::thread::spawn(move || -> anyhow::Result<Vec<HitRecord>> {
        let file = std::fs::File::create(&out_path)?;
        let mut buf = std::io::BufWriter::new(file);
        let mut records = Vec::new();
        for hit in rx {
            let salt_b = b20::salt_bytes(hit.salt);
            let mut salt_hex = vec![0u8; 64];
            b20::hex_lower(&salt_b, &mut salt_hex);
            let mut tail_hex = vec![0u8; 18];
            b20::hex_lower(&hit.tail, &mut tail_hex);
            let rec = HitRecord {
                word: hit.word,
                position: hit.pos.as_str().to_string(),
                salt: hit.salt.to_string(),
                salt_bytes32: format!("0x{}", String::from_utf8(salt_hex).unwrap()),
                tail: String::from_utf8(tail_hex).unwrap(),
                asset_address: b20::eip55(&b20::b20_address(&hit.tail, 0)),
                stablecoin_address: b20::eip55(&b20::b20_address(&hit.tail, 1)),
            };
            serde_json::to_writer(&mut buf, &rec)?;
            buf.write_all(b"\n")?;
            buf.flush()?;
            println!(
                "{:>18} {:<6} salt={:<12} {}",
                rec.word, rec.position, rec.salt, rec.asset_address
            );
            records.push(rec);
        }
        Ok(records)
    });

    let t0 = Instant::now();
    let mut handles = Vec::new();
    for wid in 0..opts.workers {
        let matcher = matcher.clone();
        let stop = stop.clone();
        let scanned = scanned.clone();
        let tx = tx.clone();
        let deployer = opts.deployer;
        let start = opts.start;
        let stride = opts.workers as u128;
        let quota = quotas.as_ref().map(|q| q[wid]);
        handles.push(std::thread::spawn(move || {
            worker(
                wid as u128,
                stride,
                deployer,
                start,
                quota,
                &matcher,
                tx,
                &scanned,
                &stop,
            )
        }));
    }
    drop(tx);

    // progress loop on the main thread
    while handles.iter().any(|h| !h.is_finished()) {
        std::thread::sleep(Duration::from_millis(1000));
        let n = scanned.load(Ordering::Relaxed);
        let secs = t0.elapsed().as_secs_f64();
        let rate = n as f64 / secs.max(1e-9);
        let interval = if p_combined > 0.0 && rate > 0.0 {
            format!(
                ", a hit every ~{}s on average",
                ((1.0 / p_combined) / rate) as u64
            )
        } else {
            String::new()
        };
        eprint!(
            "\r{} salts, {:.2} MH/s{}    ",
            words::humanize(n as f64),
            rate / 1e6,
            interval
        );
    }
    eprintln!();
    for h in handles {
        h.join().expect("worker panicked");
    }
    let records = writer.join().expect("writer panicked")?;

    let n = scanned.load(Ordering::Relaxed);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "scanned {} salts from {} in {:.0}s ({:.2} MH/s) | {} hits -> {}",
        n,
        opts.start,
        dt,
        n as f64 / dt.max(1e-9) / 1e6,
        records.len(),
        opts.out.display()
    );
    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn worker(
    wid: u128,
    stride: u128,
    deployer: [u8; 20],
    start: u128,
    per_worker: Option<u64>,
    matcher: &Matcher,
    tx: mpsc::Sender<RawHit>,
    scanned: &AtomicU64,
    stop: &AtomicBool,
) {
    let mut pre = [0u8; 64];
    pre[12..32].copy_from_slice(&deployer);
    let mut i = match start.checked_add(wid) {
        Some(v) => v,
        None => return,
    };
    let mut n: u64 = 0;
    let mut since: u64 = 0;
    loop {
        if let Some(c) = per_worker {
            if n >= c {
                break;
            }
        }
        pre[48..].copy_from_slice(&i.to_be_bytes()); // bytes 32..48 stay zero
        let mut h = [0u8; 32];
        let mut k = Keccak::v256();
        k.update(&pre);
        k.finalize(&mut h);
        let mut tail_hex = [0u8; 18];
        b20::hex_lower(&h[..9], &mut tail_hex);
        if let Some((pos, word)) = matcher.find(&tail_hex) {
            let mut tail = [0u8; 9];
            tail.copy_from_slice(&h[..9]);
            if tx
                .send(RawHit {
                    word: word.to_string(),
                    pos,
                    salt: i,
                    tail,
                })
                .is_err()
            {
                break;
            }
        }
        n += 1;
        since += 1;
        // stop flag checked every 65,536 salts (~10 ms of work), not every
        // iteration: an atomic load in the hot loop costs more than prompt
        // Ctrl-C response is worth
        if since == 65_536 {
            scanned.fetch_add(since, Ordering::Relaxed);
            since = 0;
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
        i = match i.checked_add(stride) {
            Some(v) => v,
            None => break,
        };
    }
    scanned.fetch_add(since, Ordering::Relaxed);
}
