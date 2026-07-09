use clap::ValueEnum;

#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub enum Positions {
    Prefix,
    Suffix,
    Ends,
    Any,
}

pub fn parse_words(raw: &str) -> Result<Vec<String>, String> {
    let mut words: Vec<String> = raw
        .split(',')
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    words.sort();
    words.dedup();
    if words.is_empty() {
        return Err("no words given".into());
    }
    for w in &words {
        if !w.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(format!(
                "not hex-expressible: {w} (only 0-9 a-f; leetspeak helps: o=0 l/i=1 s=5 t=7 g=6 z=2)"
            ));
        }
        if w.len() > 18 {
            return Err(format!("longer than the 18-char window: {w}"));
        }
    }
    // longest first so longer words win matching ties; stable tiebreak alphabetical
    words.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    Ok(words)
}

/// Approximate expected salts per hit for one word of length `len`.
/// Union probability over enabled placements; the search is memoryless, so
/// callers must present this as an average interval, never a countdown.
pub fn expected_salts_per_hit(len: usize, positions: Positions, inner_min: usize) -> f64 {
    let p_end = 16f64.powi(-(len as i32));
    if len == 18 {
        // prefix, suffix, and inner are all the same single placement
        return 16f64.powi(18);
    }
    let p = match positions {
        Positions::Prefix | Positions::Suffix => p_end,
        Positions::Ends | Positions::Any => {
            // P(prefix or suffix): subtract the both-ends overlap when the two
            // placements are disjoint; for 2*len > 18 the overlap term is
            // negligible (< 16^-18) and treated as 0.
            let both = if 2 * len <= 18 {
                16f64.powi(-(2 * len as i32))
            } else {
                0.0
            };
            let mut p = 2.0 * p_end - both;
            if positions == Positions::Any && len >= inner_min {
                // 19-len placements total, minus the 2 end placements
                p += (17 - len) as f64 * p_end;
            }
            p
        }
    };
    1.0 / p
}

pub fn humanize(n: f64) -> String {
    const UNITS: &[(f64, &str)] = &[(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")];
    for (scale, suffix) in UNITS {
        if n >= *scale {
            return format!("{:.1}{}", n / scale, suffix);
        }
    }
    format!("{n:.0}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_sorts_longest_first() {
        let w = parse_words("dead, C0FFEE,dead,beef").unwrap();
        assert_eq!(w, vec!["c0ffee", "beef", "dead"]);
    }

    #[test]
    fn rejects_non_hex_with_leet_hint() {
        let e = parse_words("hello").unwrap_err();
        assert!(e.contains("leetspeak"), "hint missing: {e}");
    }

    #[test]
    fn rejects_empty_and_too_long() {
        assert!(parse_words("  , ,").is_err());
        assert!(parse_words("aaaaaaaaaaaaaaaaaaa").is_err()); // 19 chars
    }

    #[test]
    fn expected_math() {
        // single end: 16^6
        let one = expected_salts_per_hit(6, Positions::Prefix, 6);
        assert!((one - 16f64.powi(6)).abs() < 1.0);
        // both ends: ~half of that (union, correction negligible at L=6)
        let ends = expected_salts_per_hit(6, Positions::Ends, 6);
        assert!((ends - 16f64.powi(6) / 2.0).abs() / ends < 0.001);
        // full window: single placement, no doubling
        assert_eq!(expected_salts_per_hit(18, Positions::Ends, 6), 16f64.powi(18));
        // any: 19 - L placements at L=8 -> 11x a single end
        let any = expected_salts_per_hit(8, Positions::Any, 6);
        assert!((any - 16f64.powi(8) / 11.0).abs() / any < 0.01);
        // any but word below inner-min: ends only
        let short = expected_salts_per_hit(4, Positions::Any, 6);
        assert!((short - 16f64.powi(4) / 2.0).abs() / short < 0.01);
    }

    #[test]
    fn humanizes() {
        assert_eq!(humanize(8_388_608.0), "8.4M");
        assert_eq!(humanize(2_100_000_000.0), "2.1B");
        assert_eq!(humanize(950.0), "950");
    }
}
