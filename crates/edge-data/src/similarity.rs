//! String similarity for entity resolution.
//!
//! Venues name the same team "Los Angeles Lakers", "LA Lakers", "L.A. Lakers",
//! and "Lakers". An exact-match index handles whatever anyone thought to write
//! down as an alias; this handles the rest.
//!
//! The metric is the normalised **indel** (insert/delete) distance, the same
//! one `rapidfuzz.fuzz.ratio` uses, so scores here are comparable to the
//! thresholds the Python implementation was tuned against. Substitutions cost
//! two rather than one, which is the property that matters for names: "Rangers"
//! and "Raiders" differ by two letters in place, and Levenshtein flatters that
//! resemblance in a way indel distance does not.
//!
//! Two token strategies sit on top, because names disagree in two distinct
//! ways:
//!
//! - **Word order** — "Inter Milan" against "Milan Inter". [`token_sort_ratio`]
//!   sorts the tokens first.
//! - **Missing words** — "Lakers" against "Los Angeles Lakers". Sorting does
//!   not help; the strings genuinely differ by two-thirds of their length.
//!   [`token_set_ratio`] compares the shared tokens against the whole, which
//!   scores a subset as a match.
//!
//! Neither strategy can bridge "LA Lakers" and "Los Angeles Lakers", which is
//! not a spelling difference at all — the tokens share no characters in the
//! order that matters. [`similarity`] therefore first collapses any run of
//! words whose initials spell a token on the other side, which turns that pair
//! into an exact one. This is the single most common way American venues
//! disagree about a team name.
//!
//! [`similarity`] takes the best of the strategies. That is deliberately
//! generous, and it is safe only because the resolver refuses to act on a
//! strong match that has a comparably strong runner-up — see [`crate::resolve`].

use std::collections::BTreeSet;

/// Tokens carrying no identifying information in team and competition names.
/// Dropping them stops "FC Barcelona" and "Barcelona" from being treated as
/// different, and stops the shared "FC" in two unrelated clubs from inflating
/// their similarity.
const NOISE: &[&str] = &["fc", "afc", "cf", "sc", "ac", "the", "club", "team"];

/// Fold to a comparable form: lowercase, de-accented, punctuation to spaces,
/// whitespace collapsed, and runs of single letters rejoined.
///
/// Punctuation becomes a *space* rather than being deleted, so "St.Louis"
/// tokenises as two words rather than the single blob `stlouis`. That alone
/// would shred "L.A. Lakers" into `l a lakers`, so consecutive single-letter
/// tokens are then glued back together — an initialism written with periods
/// and one written without must normalise to the same thing.
pub fn normalize(s: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        let folded = deaccent(ch);
        if folded.is_alphanumeric() {
            for c in folded.to_lowercase() {
                cur.push(c);
            }
        } else if !cur.is_empty() {
            parts.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        parts.push(cur);
    }

    let mut out: Vec<String> = Vec::with_capacity(parts.len());
    let mut run = String::new();
    for p in parts {
        let single = p.chars().count() == 1 && p.chars().all(char::is_alphabetic);
        if single {
            run.push_str(&p);
        } else {
            if !run.is_empty() {
                out.push(std::mem::take(&mut run));
            }
            out.push(p);
        }
    }
    if !run.is_empty() {
        out.push(run);
    }
    out.join(" ")
}

/// Latin-1 and Latin Extended-A letters folded to ASCII. Not a full Unicode
/// normalisation — just enough that "Atlético" and "Atletico" are one team.
fn deaccent(c: char) -> char {
    match c {
        'à'..='å' | 'ā' | 'ă' | 'ą' => 'a',
        'è'..='ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ì'..='ï' | 'ĩ' | 'ī' | 'į' | 'ı' => 'i',
        'ò'..='ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ù'..='ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' => 'u',
        'ç' | 'ć' | 'ĉ' | 'č' => 'c',
        'ñ' | 'ń' | 'ň' => 'n',
        'ý' | 'ÿ' => 'y',
        'š' | 'ś' => 's',
        'ž' | 'ź' | 'ż' => 'z',
        'À'..='Å' => 'A',
        'È'..='Ë' => 'E',
        'Ì'..='Ï' => 'I',
        'Ò'..='Ö' | 'Ø' => 'O',
        'Ù'..='Ü' => 'U',
        'Ç' => 'C',
        'Ñ' => 'N',
        other => other,
    }
}

/// Normalised tokens with noise words removed.
///
/// If a name is *entirely* noise ("The Club") the noise words are kept — a
/// nameless entity would match everything.
pub fn tokens(s: &str) -> Vec<String> {
    let normalized = normalize(s);
    let all: Vec<&str> = normalized.split_whitespace().collect();
    let kept: Vec<String> =
        all.iter().filter(|t| !NOISE.contains(t)).map(|t| (*t).to_string()).collect();
    if kept.is_empty() { all.into_iter().map(String::from).collect() } else { kept }
}

/// Length of the longest common subsequence, in characters.
fn lcs_len(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    // Two rows is enough; we only need the length, never the alignment.
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for &ca in a {
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = if ca == cb { prev[j] + 1 } else { cur[j].max(prev[j + 1]) };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Similarity of two already-normalised strings, in `[0, 1]`.
///
/// `2·LCS / (|a| + |b|)` — the complement of normalised indel distance. Two
/// empty strings are `1.0`; one empty string is `0.0`.
pub fn ratio(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let (ac, bc): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let total = ac.len() + bc.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * lcs_len(&ac, &bc) as f64 / total as f64
}

/// [`ratio`] after sorting each side's tokens. Insensitive to word order.
pub fn token_sort_ratio(a: &str, b: &str) -> f64 {
    let mut ta = tokens(a);
    let mut tb = tokens(b);
    ta.sort();
    tb.sort();
    ratio(&ta.join(" "), &tb.join(" "))
}

/// [`ratio`] over the shared tokens versus each full name.
///
/// This is what lets "Lakers" resolve to "Los Angeles Lakers": the shared token
/// set *is* one of the two names, so that comparison scores 1.0.
pub fn token_set_ratio(a: &str, b: &str) -> f64 {
    let sa: BTreeSet<String> = tokens(a).into_iter().collect();
    let sb: BTreeSet<String> = tokens(b).into_iter().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }

    let join = |it: &mut dyn Iterator<Item = &String>| {
        it.map(String::as_str).collect::<Vec<_>>().join(" ")
    };
    let shared = join(&mut sa.intersection(&sb));
    if shared.is_empty() {
        // Nothing in common at the token level; fall back to characters, which
        // still catches a misspelling inside a single-token name.
        return token_sort_ratio(a, b);
    }
    let only_a = join(&mut sa.difference(&sb));
    let only_b = join(&mut sb.difference(&sa));
    let full_a = format!("{shared} {only_a}").trim_end().to_string();
    let full_b = format!("{shared} {only_b}").trim_end().to_string();

    ratio(&shared, &full_a)
        .max(ratio(&shared, &full_b))
        .max(ratio(&full_a, &full_b))
}

/// Collapse runs of words in `long` whose initials spell a token of `short`.
///
/// `["los", "angeles", "lakers"]` against a name containing `la` becomes
/// `["la", "lakers"]`. Only runs of length 2–4 are considered, and only where
/// at least one collapsed word is longer than a letter, so this rewrites
/// spelled-out names towards abbreviations and never the reverse.
fn align_initialisms(short: &[String], long: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(long.len());
    let mut i = 0;
    while i < long.len() {
        let hit = short.iter().find(|t| {
            let n = t.chars().count();
            (2..=4).contains(&n)
                && i + n <= long.len()
                && long[i..i + n].iter().any(|w| w.chars().count() > 1)
                && long[i..i + n].iter().filter_map(|w| w.chars().next()).eq(t.chars())
        });
        match hit {
            Some(t) => {
                out.push(t.clone());
                i += t.chars().count();
            }
            None => {
                out.push(long[i].clone());
                i += 1;
            }
        }
    }
    out
}

/// The score entity resolution uses: the best of order-insensitive and
/// subset-tolerant matching, after reconciling abbreviations against the names
/// they abbreviate.
pub fn similarity(a: &str, b: &str) -> f64 {
    let (ta, tb) = (tokens(a), tokens(b));
    let (aa, bb) = (align_initialisms(&tb, &ta).join(" "), align_initialisms(&ta, &tb).join(" "));
    token_sort_ratio(&aa, &bb).max(token_set_ratio(&aa, &bb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_folds_case_punctuation_and_accents() {
        assert_eq!(normalize("  L.A.  Lakers "), "la lakers");
        assert_eq!(normalize("Atlético Madrid"), "atletico madrid");
        assert_eq!(normalize("Saint-Étienne"), "saint etienne");
        assert_eq!(normalize("76ers"), "76ers");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn punctuation_separates_but_an_initialism_is_put_back_together() {
        assert_eq!(tokens("St. Louis Blues"), vec!["st", "louis", "blues"]);
        // Splitting on the dots alone would give ["l", "a", "lakers"], which
        // matches neither "LA Lakers" nor "Los Angeles Lakers".
        assert_eq!(tokens("L.A. Lakers"), vec!["la", "lakers"]);
        assert_eq!(similarity("L.A. Lakers", "LA Lakers"), 1.0);
    }

    #[test]
    fn noise_words_are_dropped_but_never_all_of_them() {
        assert_eq!(tokens("FC Barcelona"), vec!["barcelona"]);
        assert_eq!(tokens("The Club"), vec!["the", "club"], "a nameless entity matches everything");
    }

    #[test]
    fn identical_strings_score_one_and_disjoint_ones_score_low() {
        assert_eq!(ratio("lakers", "lakers"), 1.0);
        assert_eq!(ratio("", ""), 1.0);
        assert_eq!(ratio("lakers", ""), 0.0);
        // Unrelated names still share letters, so raw character similarity sits
        // well above zero. Separating teams is the threshold's job, not this
        // function's — see `a_shared_city_is_not_a_shared_team`.
        assert!(ratio("lakers", "yankees") < 0.7);
    }

    #[test]
    fn a_typo_still_scores_high() {
        assert!(ratio("celtics", "celtcs") > 0.9, "{}", ratio("celtics", "celtcs"));
        assert!(ratio("manchester", "manchestr") > 0.9);
    }

    #[test]
    fn word_order_does_not_matter() {
        assert_eq!(token_sort_ratio("Inter Milan", "Milan Inter"), 1.0);
        assert!(token_sort_ratio("Inter Milan", "Milan Inter") > ratio("inter milan", "milan inter"));
    }

    #[test]
    fn a_subset_name_matches_the_full_one() {
        // The case plain sorting cannot handle: the strings differ by
        // two-thirds of their length and are still the same team.
        assert!(token_sort_ratio("Lakers", "Los Angeles Lakers") < 0.7);
        assert_eq!(token_set_ratio("Lakers", "Los Angeles Lakers"), 1.0);
        assert_eq!(similarity("Lakers", "Los Angeles Lakers"), 1.0);
    }

    #[test]
    fn a_shared_city_is_not_a_shared_team() {
        // The dangerous case for a subset-tolerant metric: two real teams that
        // share half their name. Resolving these to each other would have the
        // engine arbitrage two different games against one another.
        let s = similarity("New York Rangers", "New York Knicks");
        assert!(s < 0.85, "{s} is above the resolver's threshold");
        let s = similarity("Los Angeles Lakers", "Los Angeles Clippers");
        assert!(s < 0.85, "{s} is above the resolver's threshold");
        let s = similarity("Manchester United", "Manchester City");
        assert!(s < 0.85, "{s} is above the resolver's threshold");
    }

    #[test]
    fn substitutions_are_penalised_more_than_edits() {
        // "Rangers"/"Raiders" differ by two letters in place. Indel distance
        // charges four for that, which is the point of using it over
        // Levenshtein for names.
        assert!(ratio("rangers", "raiders") < 0.75, "{}", ratio("rangers", "raiders"));
    }

    #[test]
    fn the_metric_is_symmetric_and_bounded() {
        let names = ["Los Angeles Lakers", "Lakers", "boston celtics", "", "FC Barcelona", "76ers"];
        for a in names {
            for b in names {
                let (ab, ba) = (similarity(a, b), similarity(b, a));
                assert!((ab - ba).abs() < 1e-12, "{a:?}/{b:?}: {ab} vs {ba}");
                assert!((0.0..=1.0).contains(&ab), "{a:?}/{b:?}: {ab}");
            }
        }
    }

    #[test]
    fn names_sharing_no_tokens_still_tolerate_a_misspelling() {
        // No token intersection at all, so the set strategy has nothing to work
        // with and must not simply return zero.
        assert!(similarity("Barcelona", "Barcelna") > 0.85);
    }

    #[test]
    fn abbreviations_resolve_against_the_full_city_name() {
        // No token strategy alone gets here: "la" and "los angeles" share no
        // subsequence worth speaking of.
        assert!(token_set_ratio("LA Lakers", "Los Angeles Lakers") < 0.85);
        assert_eq!(similarity("LA Lakers", "Los Angeles Lakers"), 1.0);
        assert_eq!(similarity("NY Rangers", "New York Rangers"), 1.0);
        assert_eq!(similarity("SF Giants", "San Francisco Giants"), 1.0);
    }

    #[test]
    fn an_abbreviation_does_not_launder_a_different_team() {
        // The risk the initialism rule introduces: once "LA" and "Los Angeles"
        // are reconciled, the rest of the name must still have to agree.
        let s = similarity("LA Lakers", "Los Angeles Clippers");
        assert!(s < 0.85, "{s} is above the resolver's threshold");
    }

    #[test]
    fn initialism_folding_only_rewrites_towards_the_abbreviation() {
        // A run of genuinely single letters must not be collapsed against an
        // unrelated short word.
        assert_eq!(similarity("A B", "AB"), 1.0);
        let s = similarity("US Open", "United States Open");
        assert!(s > 0.85, "{s}");
    }
}
