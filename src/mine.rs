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
    General { words: Vec<String>, inner: Vec<String> },
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
            words.iter().filter(|w| w.len() >= inner_min).cloned().collect()
        } else {
            Vec::new()
        };
        let lens: std::collections::HashSet<usize> = words.iter().map(|w| w.len()).collect();
        let kind = if lens.len() == 1 && inner.is_empty() {
            let mut sorted = words.to_vec();
            sorted.sort();
            Kind::Uniform { len: *lens.iter().next().unwrap(), sorted }
        } else {
            Kind::General { words: words.to_vec(), inner }
        };
        Matcher { kind, check_prefix, check_suffix }
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
        assert_eq!(m.find(&t("dead12345678901234")), Some((Pos::Prefix, "dead")));
        assert_eq!(m.find(&t("12345678901234beef")), Some((Pos::Suffix, "beef")));
        assert_eq!(m.find(&t("123456dead89012345")), None); // inner not enabled
        assert_eq!(m.find(&t("123456789012345678")), None);
    }

    #[test]
    fn prefix_beats_suffix_beats_inner() {
        let m = Matcher::new(&["dead".into()], Positions::Ends, 4);
        // both ends match: prefix wins
        assert_eq!(m.find(&t("dead1234567890dead")), Some((Pos::Prefix, "dead")));
        let m = Matcher::new(&["dead".into(), "c0ffee".into()], Positions::Any, 4);
        // suffix and inner both present: suffix wins
        assert_eq!(m.find(&t("12c0ffee901234dead")), Some((Pos::Suffix, "dead")));
    }

    #[test]
    fn longest_word_wins_at_same_end() {
        let m = Matcher::new(&["deadbeef".into(), "dead".into()], Positions::Prefix, 6);
        assert_eq!(m.find(&t("deadbeef9012345678")), Some((Pos::Prefix, "deadbeef")));
    }

    #[test]
    fn inner_respects_inner_min() {
        let m = Matcher::new(&["c0ffee".into(), "dead".into()], Positions::Any, 6);
        assert_eq!(m.find(&t("12c0ffee9012345678")), Some((Pos::Inner, "c0ffee")));
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
