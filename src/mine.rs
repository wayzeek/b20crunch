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

/// One same-length run of words anchored at one end of the window: a single
/// mask selects the anchored nibbles and a sorted list of masked values is
/// binary-searched per candidate.
struct Group {
    mask: u128,
    /// (masked value, index into Matcher::words), sorted by value.
    entries: Vec<(u128, usize)>,
}

/// All window positions of one inner-eligible word as (mask, value) pairs.
struct InnerWord {
    idx: usize,
    positions: Vec<(u128, u128)>,
}

/// Matches candidate windows against the word list without leaving the
/// integer domain: the 18-nibble tail sits in bits 127..=56 of the window
/// (the kernel's big-endian view of the first 16 digest bytes), so every
/// placement of every word is one masked compare. Precedence is prefix, then
/// suffix, then inner, longest word first, exactly like the original
/// string matcher; groups are ordered longest-first and words of equal
/// length are mutually exclusive under the same mask, so a binary search
/// inside a group cannot change which word wins.
pub struct Matcher {
    /// From words::parse_words: lowercase, deduped, longest-first.
    words: Vec<String>,
    prefix: Vec<Group>,
    suffix: Vec<Group>,
    inner: Vec<InnerWord>,
}

/// Nibble count of the grindable window.
const WINDOW: usize = 18;

fn word_bits(w: &str) -> u128 {
    u128::from_str_radix(w, 16).expect("words are validated hex")
}

/// (mask, value) for a word covering nibbles p..p+len of the window.
fn placement(w: &str, p: usize) -> (u128, u128) {
    let shift = 128 - 4 * (p + w.len()) as u32;
    let mask = ((1u128 << (4 * w.len())) - 1) << shift;
    (mask, word_bits(w) << shift)
}

impl Matcher {
    /// `words` must come from words::parse_words (lowercase, longest-first).
    pub fn new(words: &[String], positions: Positions, inner_min: usize) -> Matcher {
        let (check_prefix, check_suffix) = match positions {
            Positions::Prefix => (true, false),
            Positions::Suffix => (false, true),
            Positions::Ends | Positions::Any => (true, true),
        };

        let anchored = |at_start: bool| -> Vec<Group> {
            let mut groups: Vec<Group> = Vec::new();
            for (idx, w) in words.iter().enumerate() {
                let p = if at_start { 0 } else { WINDOW - w.len() };
                let (mask, value) = placement(w, p);
                match groups.last_mut() {
                    // words arrive longest-first, so equal lengths are adjacent
                    Some(g) if g.mask == mask => g.entries.push((value, idx)),
                    _ => groups.push(Group {
                        mask,
                        entries: vec![(value, idx)],
                    }),
                }
            }
            for g in &mut groups {
                g.entries.sort_unstable();
            }
            groups
        };

        let inner: Vec<InnerWord> = if positions == Positions::Any {
            words
                .iter()
                .enumerate()
                .filter(|(_, w)| w.len() >= inner_min)
                .map(|(idx, w)| InnerWord {
                    idx,
                    positions: (0..=WINDOW - w.len()).map(|p| placement(w, p)).collect(),
                })
                .collect()
        } else {
            Vec::new()
        };

        Matcher {
            words: words.to_vec(),
            prefix: if check_prefix {
                anchored(true)
            } else {
                Vec::new()
            },
            suffix: if check_suffix {
                anchored(false)
            } else {
                Vec::new()
            },
            inner,
        }
    }

    /// At most one hit per window: prefix, then suffix, then inner.
    #[inline]
    pub fn find(&self, window: u128) -> Option<(Pos, &str)> {
        for (groups, pos) in [(&self.prefix, Pos::Prefix), (&self.suffix, Pos::Suffix)] {
            for g in groups {
                let v = window & g.mask;
                if let Ok(k) = g.entries.binary_search_by(|e| e.0.cmp(&v)) {
                    return Some((pos, &self.words[g.entries[k].1]));
                }
            }
        }
        for iw in &self.inner {
            if iw.positions.iter().any(|&(m, v)| window & m == v) {
                return Some((Pos::Inner, &self.words[iw.idx]));
            }
        }
        None
    }
}

/// One row of the GPU match table: a (mask, value) placement pre-split into
/// 64-bit halves for the kernel, tagged with its word and position class.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuEntry {
    pub mask_hi: u64,
    pub mask_lo: u64,
    pub value_hi: u64,
    pub value_lo: u64,
    pub word: u32,
    pub pos: u32,
}

#[cfg(feature = "gpu")]
impl Pos {
    /// Stable wire code for the GPU table: 0 prefix, 1 suffix, 2 inner.
    pub fn code(self) -> u32 {
        match self {
            Pos::Prefix => 0,
            Pos::Suffix => 1,
            Pos::Inner => 2,
        }
    }
}

#[cfg(feature = "gpu")]
impl Matcher {
    /// Flat match table for the GPU kernel, position-major (all prefix
    /// entries, then suffix, then inner) and longest-word-first within each
    /// class: exactly find()'s precedence, so a first-match scan in table
    /// order reproduces the CPU's winner on overlapping matches.
    pub fn gpu_entries(&self) -> Vec<GpuEntry> {
        let split = |m: u128, v: u128, word: usize, pos: u32| GpuEntry {
            mask_hi: (m >> 64) as u64,
            mask_lo: m as u64,
            value_hi: (v >> 64) as u64,
            value_lo: v as u64,
            word: word as u32,
            pos,
        };
        let mut t = Vec::new();
        for (groups, pos) in [(&self.prefix, 0u32), (&self.suffix, 1)] {
            for g in groups {
                for &(v, idx) in &g.entries {
                    t.push(split(g.mask, v, idx, pos));
                }
            }
        }
        for iw in &self.inner {
            for &(m, v) in &iw.positions {
                t.push(split(m, v, iw.idx, 2));
            }
        }
        t
    }

    /// The word behind a GpuEntry::word index; None flags a corrupt record.
    pub fn word(&self, idx: usize) -> Option<&str> {
        self.words.get(idx).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::words::Positions;

    /// 18 hex chars -> the window u128 the kernel would produce (junk bits zero).
    fn t(s: &str) -> u128 {
        assert_eq!(s.len(), 18);
        u128::from_str_radix(s, 16).unwrap() << (128 - 4 * WINDOW as u32)
    }

    #[test]
    fn uniform_fast_path_prefix_and_suffix() {
        let m = Matcher::new(&["dead".into(), "beef".into()], Positions::Ends, 6);
        assert_eq!(m.find(t("dead12345678901234")), Some((Pos::Prefix, "dead")));
        assert_eq!(m.find(t("12345678901234beef")), Some((Pos::Suffix, "beef")));
        assert_eq!(m.find(t("123456dead89012345")), None); // inner not enabled
        assert_eq!(m.find(t("123456789012345678")), None);
    }

    #[test]
    fn prefix_beats_suffix_beats_inner() {
        let m = Matcher::new(&["dead".into()], Positions::Ends, 4);
        // both ends match: prefix wins
        assert_eq!(m.find(t("dead1234567890dead")), Some((Pos::Prefix, "dead")));
        let m = Matcher::new(&["dead".into(), "c0ffee".into()], Positions::Any, 4);
        // suffix and inner both present: suffix wins
        assert_eq!(m.find(t("12c0ffee901234dead")), Some((Pos::Suffix, "dead")));
    }

    #[test]
    fn longest_word_wins_at_same_end() {
        let m = Matcher::new(&["deadbeef".into(), "dead".into()], Positions::Prefix, 6);
        assert_eq!(
            m.find(t("deadbeef9012345678")),
            Some((Pos::Prefix, "deadbeef"))
        );
    }

    #[test]
    fn inner_respects_inner_min() {
        let m = Matcher::new(&["c0ffee".into(), "dead".into()], Positions::Any, 6);
        assert_eq!(
            m.find(t("12c0ffee9012345678")),
            Some((Pos::Inner, "c0ffee"))
        );
        assert_eq!(m.find(t("12dead345678901234")), None); // 4 < inner_min 6
    }

    #[test]
    fn prefix_only_and_suffix_only() {
        let m = Matcher::new(&["dead".into()], Positions::Prefix, 6);
        assert_eq!(m.find(t("12345678901234dead")), None);
        let m = Matcher::new(&["dead".into()], Positions::Suffix, 6);
        assert_eq!(m.find(t("dead12345678901234")), None);
    }

    #[test]
    fn window_filling_words_match_at_the_boundaries() {
        // 18 chars: prefix, suffix, and inner are the same single placement
        let full = "abcdef0123456789ab";
        let m = Matcher::new(&[full.into()], Positions::Ends, 6);
        assert_eq!(m.find(t(full)), Some((Pos::Prefix, full)));
        assert_eq!(m.find(t("abcdef0123456789aa")), None);
        // 17 chars: the two end placements overlap in the middle 16
        let w17 = &full[..17];
        let m = Matcher::new(&[w17.to_string()], Positions::Any, 6);
        assert_eq!(m.find(t(&format!("{w17}f"))), Some((Pos::Prefix, w17)));
        assert_eq!(m.find(t(&format!("f{w17}"))), Some((Pos::Suffix, w17)));
    }

    /// The original string semantics, kept as the oracle for the mask matcher.
    fn reference_find<'a>(
        words: &'a [String],
        positions: Positions,
        inner_min: usize,
        tail: &str,
    ) -> Option<(Pos, &'a str)> {
        let (cp, cs) = match positions {
            Positions::Prefix => (true, false),
            Positions::Suffix => (false, true),
            Positions::Ends | Positions::Any => (true, true),
        };
        if cp {
            if let Some(w) = words.iter().find(|w| tail.starts_with(w.as_str())) {
                return Some((Pos::Prefix, w));
            }
        }
        if cs {
            if let Some(w) = words.iter().find(|w| tail.ends_with(w.as_str())) {
                return Some((Pos::Suffix, w));
            }
        }
        if positions == Positions::Any {
            if let Some(w) = words
                .iter()
                .find(|w| w.len() >= inner_min && tail.contains(w.as_str()))
            {
                return Some((Pos::Inner, w));
            }
        }
        None
    }

    #[test]
    fn matches_reference_on_random_words_and_tails() {
        let mut state = 0x2545f4914f6cdd1du128;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let hex = |n: u128, len: usize| format!("{n:018x}")[..len].to_string();
        for _ in 0..300 {
            // 1..=4 random words, lengths 1..=8 so collisions actually happen
            let n_words = 1 + (rng() % 4) as usize;
            let raw: Vec<String> = (0..n_words)
                .map(|_| hex(rng(), 1 + (rng() % 8) as usize))
                .collect();
            let words = crate::words::parse_words(&raw.join(",")).unwrap();
            for positions in [
                Positions::Prefix,
                Positions::Suffix,
                Positions::Ends,
                Positions::Any,
            ] {
                for inner_min in [2, 6] {
                    let m = Matcher::new(&words, positions, inner_min);
                    for _ in 0..40 {
                        // half the tails get a word planted at a random spot
                        let mut tail = format!("{:018x}", rng() >> 56);
                        if rng() % 2 == 0 {
                            let w = &words[(rng() % words.len() as u128) as usize];
                            let p = (rng() % (19 - w.len() as u128)) as usize;
                            tail.replace_range(p..p + w.len(), w);
                        }
                        assert_eq!(
                            m.find(t(&tail)),
                            reference_find(&words, positions, inner_min, &tail),
                            "words {words:?} positions {positions:?} inner_min {inner_min} tail {tail}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_table_is_position_major_longest_first() {
        let words = crate::words::parse_words("ab,c").unwrap(); // ["ab", "c"]
        let m = Matcher::new(&words, Positions::Any, 1);
        let t = m.gpu_entries();
        // prefix ab, prefix c, suffix ab, suffix c, inner ab (17 placements),
        // inner c (18 placements)
        assert_eq!(t.len(), 2 + 2 + 17 + 18);
        let pos_seq: Vec<u32> = t.iter().map(|e| e.pos).collect();
        assert_eq!(&pos_seq[..4], &[0, 0, 1, 1]);
        assert!(pos_seq[4..].iter().all(|&p| p == 2));
        // longest word first within each class
        assert_eq!((t[0].word, t[1].word), (0, 1));
        assert_eq!((t[2].word, t[3].word), (0, 1));
        assert_eq!(t[4].word, 0);
        assert_eq!(t[4 + 17].word, 1);
        // "ab" as prefix masks the top byte of the window
        assert_eq!(t[0].mask_hi, 0xFF00_0000_0000_0000);
        assert_eq!(t[0].mask_lo, 0);
        assert_eq!(t[0].value_hi, 0xab00_0000_0000_0000);
    }
}

use crate::{b20, kernel, words};
use anyhow::Context;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

pub struct MineOpts {
    pub deployer: [u8; 20],
    pub words: Vec<String>,
    pub positions: words::Positions,
    pub inner_min: usize,
    pub start: u128,
    pub count: Option<u64>,
    pub backend: Backend,
    pub out: PathBuf,
}

/// Which producer fills the hit channel; everything downstream (writer,
/// progress, Ctrl-C, summary) is backend-agnostic.
pub enum Backend {
    Cpu { workers: usize },
    Gpu(GpuConfig),
}

/// GPU knobs. `device`/`batch` come from the CLI; `capacity` (hit records
/// per batch) is internal, overridden by tests to force the overflow path.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    pub device: Option<usize>,
    pub batch: Option<u64>,
    pub capacity: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            device: None,
            batch: None,
            capacity: 1 << 16,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
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
    match &opts.backend {
        Backend::Cpu { workers } => {
            anyhow::ensure!(*workers >= 1, "--workers must be at least 1")
        }
        // GPU support lands behind the `gpu` Cargo feature in a later change;
        // fail before any file or thread work when it is requested here.
        Backend::Gpu(_) => {
            anyhow::bail!("this binary was built without GPU support; rebuild with --features gpu")
        }
    }
    let matcher = Arc::new(Matcher::new(&opts.words, opts.positions, opts.inner_min));
    let stop = Arc::new(AtomicBool::new(false));
    let scanned = Arc::new(AtomicU64::new(0));
    // ctrlc handler may already be installed when run() is called twice in one
    // process (tests); ignore the AlreadyInstalled error.
    // Contract: first Ctrl-C requests a graceful stop of mining; a second
    // Ctrl-C exits immediately with 130, the conventional SIGINT code. Once
    // mining itself has concluded (below, whether by finishing cleanly or by
    // Ctrl-C), `stop` is forced true, so a Ctrl-C during the subsequent
    // `--verify` network phase always hard-exits on its first press instead
    // of requiring a second one -- the escape hatch for a hung RPC call.
    {
        let stop = stop.clone();
        let _ = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::Relaxed) {
                std::process::exit(130);
            }
        });
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

    // Workers pull chunks from a shared offset counter instead of taking
    // fixed quotas: on asymmetric cores (Apple Silicon P/E) fixed splits
    // leave the fast cores idle while the slow ones finish their share, and
    // dynamic dispatch also keeps the scanned region one contiguous range.
    let next = Arc::new(AtomicU64::new(0));
    let (tx, rx) = mpsc::channel::<RawHit>();

    // Open the output file up front so a bad path fails immediately with a
    // clear error, instead of after the workers have already burned CPU (an
    // unbounded run would otherwise mine indefinitely against a dead writer).
    // Append-only so a resumed run (`--start`) accumulates hits on top of a
    // prior run's file instead of truncating it.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&opts.out)
        .with_context(|| format!("cannot open output file {}", opts.out.display()))?;

    // writer thread: JSONL file + console line per hit
    let writer = std::thread::spawn(move || -> anyhow::Result<Vec<HitRecord>> {
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
    let handles: Vec<std::thread::JoinHandle<anyhow::Result<()>>> = match &opts.backend {
        Backend::Cpu { workers } => (0..*workers)
            .map(|_| {
                let matcher = matcher.clone();
                let stop = stop.clone();
                let scanned = scanned.clone();
                let next = next.clone();
                let tx = tx.clone();
                let deployer = opts.deployer;
                let start = opts.start;
                let count = opts.count;
                std::thread::spawn(move || {
                    worker(deployer, start, count, &next, &matcher, tx, &scanned, &stop);
                    Ok(())
                })
            })
            .collect(),
        Backend::Gpu(_) => unreachable!("rejected at the top of run() until the gpu feature lands"),
    };
    drop(tx);

    // progress loop on the main thread: poll often so run end is noticed
    // within ~50ms (a full-second wait would bleed into the measured rate
    // of short runs), but keep the once-a-second print cadence
    let mut last_print = Instant::now();
    while handles.iter().any(|h| !h.is_finished()) {
        std::thread::sleep(Duration::from_millis(50));
        if last_print.elapsed() < Duration::from_secs(1) {
            continue;
        }
        last_print = Instant::now();
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
    // Mining is over here, whether it ran to completion or was Ctrl-C'd.
    // Force `stop` true so a Ctrl-C during the subsequent `--verify` network
    // phase always sees it already true and hard-exits on the very first
    // press, even when mining itself finished cleanly and never set it.
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("producer panicked")?;
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

/// Salts per dispatch chunk: big enough that the shared counter is
/// uncontended, small enough that Ctrl-C lands within a few tens of
/// milliseconds and the end-of-run straggler window stays negligible.
const CHUNK: u64 = 1 << 17;

// The dispatch counter is a u64 salt offset, so a single unbounded (`count ==
// None`) invocation covers offsets 0..2^64 before it saturates and workers
// exit -- roughly 5000 years at present throughput. Resuming past that with
// `--start` reaches the rest of the u128 salt space; a wider counter would
// only add contention to the hot dispatch path for a horizon no run reaches.

#[allow(clippy::too_many_arguments)]
fn worker(
    deployer: [u8; 20],
    start: u128,
    count: Option<u64>,
    next: &AtomicU64,
    matcher: &Matcher,
    tx: mpsc::Sender<RawHit>,
    scanned: &AtomicU64,
    stop: &AtomicBool,
) {
    let kernel = kernel::TailKernel::new(&deployer);
    let send_hit = |salt: u128, window: u128, pos: Pos, word: &str| -> bool {
        let mut tail = [0u8; 9];
        tail.copy_from_slice(&window.to_be_bytes()[..9]);
        tx.send(RawHit {
            word: word.to_string(),
            pos,
            salt,
            tail,
        })
        .is_ok()
    };
    loop {
        // saturating grab: a plain fetch_add would wrap the shared counter at
        // u64::MAX and hand already-scanned ranges out again
        let begin = next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_add(CHUNK))
            })
            .expect("fetch_update closure always returns Some");
        let end = match count {
            Some(c) => c.min(begin.saturating_add(CHUNK)),
            None => begin.saturating_add(CHUNK),
        };
        if begin >= end {
            break;
        }
        let Some(first) = start.checked_add(begin as u128) else {
            break;
        };
        // clamp so no salt in the chunk can wrap past u128::MAX
        let len = (((end - begin) as u128 - 1).min(u128::MAX - first) + 1) as u64;

        let mut off: u64 = 0;
        while off + 1 < len {
            let (sa, sb) = (first + off as u128, first + off as u128 + 1);
            let (wa, wb) = kernel.window2(sa, sb);
            if let Some((pos, word)) = matcher.find(wa) {
                if !send_hit(sa, wa, pos, word) {
                    return;
                }
            }
            if let Some((pos, word)) = matcher.find(wb) {
                if !send_hit(sb, wb, pos, word) {
                    return;
                }
            }
            off += 2;
        }
        if off < len {
            let sa = first + off as u128;
            let wa = kernel.window(sa);
            if let Some((pos, word)) = matcher.find(wa) {
                if !send_hit(sa, wa, pos, word) {
                    return;
                }
            }
        }

        scanned.fetch_add(len, Ordering::Relaxed);
        // one stop check per chunk (~131k salts): prompt Ctrl-C without an
        // atomic load in the hash loop; a clamped chunk means the u128 salt
        // space itself is exhausted
        if stop.load(Ordering::Relaxed) || len < end - begin {
            break;
        }
    }
}
