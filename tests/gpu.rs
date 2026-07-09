#![cfg(feature = "gpu")]
use b20crunch::{b20, mine, words};
use std::collections::HashSet;

const DEPLOYER: &str = "0x1111111111111111111111111111111111111111";

fn run_backend(
    backend: mine::Backend,
    word_list: &str,
    positions: words::Positions,
    inner_min: usize,
    start: u128,
    count: u64,
) -> (Vec<mine::HitRecord>, usize) {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("hits.jsonl");
    let hits = mine::run(mine::MineOpts {
        deployer: b20::parse_address(DEPLOYER).unwrap(),
        words: words::parse_words(word_list).unwrap(),
        positions,
        inner_min,
        start,
        count: Some(count),
        backend,
        out: out.clone(),
    })
    .unwrap();
    let lines = std::fs::read_to_string(&out).unwrap().lines().count();
    (hits, lines)
}

/// (salt, word, position): the identity of a hit, order-independent.
fn key_set(hits: &[mine::HitRecord]) -> HashSet<(String, String, String)> {
    hits.iter()
        .map(|h| (h.salt.clone(), h.word.clone(), h.position.clone()))
        .collect()
}

fn cpu() -> mine::Backend {
    mine::Backend::Cpu { workers: 2 }
}

fn gpu() -> mine::Backend {
    mine::Backend::Gpu(mine::GpuConfig::default())
}

#[test]
fn hit_sets_match_on_a_fixed_range() {
    let (c, _) = run_backend(cpu(), "dead,beef", words::Positions::Ends, 6, 0, 4_000_000);
    let (g, glines) = run_backend(gpu(), "dead,beef", words::Positions::Ends, 6, 0, 4_000_000);
    assert!(!c.is_empty());
    assert_eq!(key_set(&c), key_set(&g));
    assert_eq!(glines, g.len(), "GPU run wrote duplicate JSONL lines");
}

#[test]
fn precedence_matches_cpu_on_overlapping_words() {
    // 2-nibble words firing in every class on most salts: any divergence in
    // prefix > suffix > inner or longest-first ordering breaks set equality
    let (c, _) = run_backend(cpu(), "de,ad", words::Positions::Any, 2, 0, 300_000);
    let (g, _) = run_backend(gpu(), "de,ad", words::Positions::Any, 2, 0, 300_000);
    assert!(c.len() > 10_000, "overlap range too sparse: {}", c.len());
    // the range must actually contain contested tails (two or more position
    // classes matching at once), otherwise this equality pins precedence
    // vacuously; deterministic for the fixed deployer and range
    let contested = c
        .iter()
        .filter(|h| {
            let t = &h.tail; // 18 ascii hex chars
            let classes = [
                t.starts_with("de") || t.starts_with("ad"),
                t.ends_with("de") || t.ends_with("ad"),
                // chars 1..=16: exactly the inner placements for 2-char words
                t[1..17].contains("de") || t[1..17].contains("ad"),
            ];
            classes.iter().filter(|&&x| x).count() >= 2
        })
        .count();
    assert!(
        contested > 50,
        "too few contested tails to pin precedence: {contested}"
    );
    assert_eq!(key_set(&c), key_set(&g));
}

#[test]
fn boundary_range_crosses_the_u64_carry() {
    let start = (u64::MAX as u128) - 500_000;
    let (c, _) = run_backend(cpu(), "dead", words::Positions::Ends, 6, start, 1_000_000);
    let (g, _) = run_backend(gpu(), "dead", words::Positions::Ends, 6, start, 1_000_000);
    assert_eq!(key_set(&c), key_set(&g));
}

#[test]
fn range_end_clamps_at_u128_max() {
    // both backends must stop at u128::MAX instead of wrapping
    let start = u128::MAX - 100_000;
    let (c, _) = run_backend(cpu(), "dead", words::Positions::Ends, 6, start, 1_000_000);
    let (g, _) = run_backend(gpu(), "dead", words::Positions::Ends, 6, start, 1_000_000);
    assert_eq!(key_set(&c), key_set(&g));
}

#[test]
fn forced_overflow_shrinks_and_loses_nothing() {
    // ~1/16 of salts hit; capacity 8 against 64k batches forces the
    // shrink-and-rerun path over and over
    let over = mine::Backend::Gpu(mine::GpuConfig {
        device: None,
        batch: Some(1 << 16),
        capacity: 8,
    });
    let (c, _) = run_backend(cpu(), "7", words::Positions::Prefix, 6, 0, 400_000);
    let (g, glines) = run_backend(over, "7", words::Positions::Prefix, 6, 0, 400_000);
    assert!(c.len() > 20_000, "hit-dense range too sparse: {}", c.len());
    assert_eq!(key_set(&c), key_set(&g));
    assert_eq!(glines, g.len(), "overflow rerun duplicated JSONL hits");
}
