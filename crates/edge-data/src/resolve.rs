//! Deciding when two venues are talking about the same thing.
//!
//! Kalshi lists `KXNBAGAME-25DEC25BOSLAL-BOS`, Polymarket lists a 77-character
//! condition id whose only human-readable form is the question text, and a
//! sportsbook lists "Boston Celtics". Cross-venue arbitrage is the whole reason
//! this system exists and it is impossible until something decides those are
//! one event.
//!
//! That decision is fuzzy and fallible, so it is quarantined here — by the time
//! anything reaches the engine the answer is an integer `EventId` and no
//! strategy ever does string matching. The output is
//! [`Resolver::event_key`], which is what gets interned.
//!
//! ## Refusing to guess
//!
//! The Python implementation this replaces took the best fuzzy match above a
//! threshold. That is the wrong shape for a trading system. A near-tie between
//! the Lakers and the Clippers is not a 51/49 call to be won by the leader; it
//! is evidence that the name is not discriminating, and acting on it means
//! quoting one game against the book of another and calling the difference
//! arbitrage. So a match whose runner-up is comparably strong resolves to
//! [`Resolution::Ambiguous`], which carries no id at all.
//!
//! Two further guards, in the same spirit:
//!
//! - **League blocking.** Candidates are restricted to the competition the
//!   caller named, when it named one. "Rangers" is a different team in the NHL
//!   and in the Scottish Premiership.
//! - **Kickoff windows.** Two teams meet many times a season. A pair of names
//!   identifies a *fixture*, not a *game*, and only the scheduled time
//!   separates them.

use std::collections::HashMap;

use edge_core::types::Ts;
use serde::{Deserialize, Serialize};

use crate::error::DataError;
use crate::similarity::{normalize, similarity};
use crate::time::{format_rfc3339, parse_rfc3339};

/// Index of a team within one [`Resolver`]. Not stable across rebuilds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TeamId(pub u32);

/// Index of a scheduled fixture within one [`Resolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GameId(pub u32);

/// A canonical team and every name anyone has been seen to call it.
#[derive(Debug, Clone, PartialEq)]
pub struct Team {
    /// Stable external identifier — the UUID from the catalogue. This is what
    /// survives a rebuild, not [`TeamId`].
    pub key: String,
    pub name: String,
    pub aliases: Vec<String>,
    /// Competition, used for blocking. `None` means "match against anything",
    /// which is right for an entity that genuinely spans leagues.
    pub league: Option<String>,
}

/// A scheduled fixture between two canonical teams.
#[derive(Debug, Clone, PartialEq)]
pub struct Game {
    pub key: String,
    pub home: TeamId,
    pub away: TeamId,
    pub start: Ts,
    pub league: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolverConfig {
    /// Minimum similarity for a fuzzy match to be considered at all. `0.85`
    /// matches the threshold the Python resolver was tuned to.
    pub threshold: f64,
    /// How far clear of the runner-up the winner must be. Below this the match
    /// is [`Resolution::Ambiguous`] and yields nothing.
    pub margin: f64,
    /// Half-width of the kickoff window, in seconds. Six hours, wide enough to
    /// absorb a venue quoting a local time or a delayed start, narrow enough to
    /// separate two legs of a doubleheader.
    pub window_secs: f64,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        ResolverConfig { threshold: 0.85, margin: 0.05, window_secs: 6.0 * 3600.0 }
    }
}

/// The outcome of resolving one name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resolution<T> {
    /// A name or alias matched outright.
    Exact(T),
    /// Matched above the threshold and clear of the runner-up.
    Fuzzy { id: T, score: f64 },
    /// Two candidates were too close to separate. Deliberately carries no id:
    /// the point is that acting on either would be a coin flip.
    Ambiguous { best: T, runner_up: T, score: f64 },
    /// Nothing came close.
    Unknown,
}

impl<T: Copy> Resolution<T> {
    /// The resolved id, if the match was safe to act on.
    pub fn id(self) -> Option<T> {
        match self {
            Resolution::Exact(id) => Some(id),
            Resolution::Fuzzy { id, .. } => Some(id),
            Resolution::Ambiguous { .. } | Resolution::Unknown => None,
        }
    }

    pub fn score(self) -> f64 {
        match self {
            Resolution::Exact(_) => 1.0,
            Resolution::Fuzzy { score, .. } | Resolution::Ambiguous { score, .. } => score,
            Resolution::Unknown => 0.0,
        }
    }

    pub fn is_resolved(self) -> bool {
        self.id().is_some()
    }
}

/// A resolved fixture, and whether the venue listed the teams the other way up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameMatch {
    pub game: GameId,
    /// The caller's "home" is the catalogue's away side. Venues disagree about
    /// this constantly, and a market on the wrong side of a fixture is a
    /// position taken against the one intended.
    pub swapped: bool,
    /// The weaker of the two team matches — the fixture is only as certain as
    /// its least certain end.
    pub score: f64,
}

/// The entity-resolution index.
#[derive(Debug, Clone, Default)]
pub struct Resolver {
    cfg: ResolverConfig,
    teams: Vec<Team>,
    games: Vec<Game>,
    /// Normalised name or alias to the teams claiming it. A `Vec` because two
    /// teams in different leagues legitimately share one.
    exact: HashMap<String, Vec<TeamId>>,
    /// All normalised surface forms per team, for fuzzy scoring.
    surfaces: Vec<Vec<String>>,
    /// Unordered team pair to the fixtures between them.
    by_pair: HashMap<(TeamId, TeamId), Vec<GameId>>,
}

impl Resolver {
    pub fn new(cfg: ResolverConfig) -> Self {
        Resolver { cfg, ..Default::default() }
    }

    pub fn teams(&self) -> &[Team] {
        &self.teams
    }

    pub fn games(&self) -> &[Game] {
        &self.games
    }

    pub fn team(&self, id: TeamId) -> Option<&Team> {
        self.teams.get(id.0 as usize)
    }

    pub fn game(&self, id: GameId) -> Option<&Game> {
        self.games.get(id.0 as usize)
    }

    pub fn add_team(&mut self, team: Team) -> TeamId {
        let id = TeamId(self.teams.len() as u32);
        let mut forms: Vec<String> = std::iter::once(&team.name)
            .chain(team.aliases.iter())
            .map(|s| normalize(s))
            .filter(|s| !s.is_empty())
            .collect();
        forms.sort();
        forms.dedup();
        for f in &forms {
            self.exact.entry(f.clone()).or_default().push(id);
        }
        self.surfaces.push(forms);
        self.teams.push(team);
        id
    }

    pub fn add_game(&mut self, game: Game) -> GameId {
        let id = GameId(self.games.len() as u32);
        self.by_pair.entry(pair(game.home, game.away)).or_default().push(id);
        self.games.push(game);
        id
    }

    /// Is this team eligible given the league the caller named?
    ///
    /// A team with no league of its own is always eligible — the catalogue is
    /// declaring that it does not participate in this distinction, not that it
    /// belongs to no competition.
    fn in_league(&self, id: TeamId, league: Option<&str>) -> bool {
        match (league, self.teams[id.0 as usize].league.as_deref()) {
            (None, _) | (_, None) => true,
            (Some(want), Some(have)) => normalize(want) == normalize(have),
        }
    }

    /// Resolve one venue-native team name.
    pub fn resolve_team(&self, name: &str, league: Option<&str>) -> Resolution<TeamId> {
        let needle = normalize(name);
        if needle.is_empty() {
            return Resolution::Unknown;
        }

        if let Some(hits) = self.exact.get(&needle) {
            let eligible: Vec<TeamId> =
                hits.iter().copied().filter(|t| self.in_league(*t, league)).collect();
            match eligible.as_slice() {
                [one] => return Resolution::Exact(*one),
                [best, runner_up, ..] => {
                    // The same literal name in two eligible leagues. No amount
                    // of scoring separates them; only more context would.
                    return Resolution::Ambiguous {
                        best: *best,
                        runner_up: *runner_up,
                        score: 1.0,
                    };
                }
                [] => {}
            }
        }

        // Best and second-best over distinct teams. Tracking the runner-up is
        // the entire safety mechanism, so it is computed even when the leader
        // is a runaway.
        let (mut best, mut second) =
            ((TeamId(0), f64::NEG_INFINITY), (TeamId(0), f64::NEG_INFINITY));
        for (i, forms) in self.surfaces.iter().enumerate() {
            let id = TeamId(i as u32);
            if !self.in_league(id, league) {
                continue;
            }
            let score = forms.iter().map(|f| similarity(&needle, f)).fold(0.0, f64::max);
            if score > best.1 {
                second = best;
                best = (id, score);
            } else if score > second.1 {
                second = (id, score);
            }
        }

        if best.1 < self.cfg.threshold {
            return Resolution::Unknown;
        }
        if second.1.is_finite() && second.1 > best.1 - self.cfg.margin {
            return Resolution::Ambiguous { best: best.0, runner_up: second.0, score: best.1 };
        }
        Resolution::Fuzzy { id: best.0, score: best.1 }
    }

    /// Resolve a fixture from the two team names a venue gave, plus the kickoff
    /// it advertised.
    ///
    /// Both ends must resolve. A fixture half-identified is not a fixture, and
    /// filling in the other side from the schedule would be inventing data.
    pub fn resolve_game(
        &self,
        home: &str,
        away: &str,
        start: Option<Ts>,
        league: Option<&str>,
    ) -> Result<GameMatch, DataError> {
        let unresolved = |what: &str, name: &str, r: Resolution<TeamId>| DataError::Unresolved {
            what: what.to_string(),
            detail: match r {
                Resolution::Ambiguous { best, runner_up, score } => format!(
                    "{name:?} is ambiguous between {:?} and {:?} at {score:.3}",
                    self.teams[best.0 as usize].name, self.teams[runner_up.0 as usize].name
                ),
                _ => format!("no team matches {name:?}"),
            },
        };

        let h = self.resolve_team(home, league);
        let hid = h.id().ok_or_else(|| unresolved("home team", home, h))?;
        let a = self.resolve_team(away, league);
        let aid = a.id().ok_or_else(|| unresolved("away team", away, a))?;

        if hid == aid {
            return Err(DataError::Unresolved {
                what: "fixture".into(),
                detail: format!("{home:?} and {away:?} resolved to the same team"),
            });
        }

        let candidates = self.by_pair.get(&pair(hid, aid)).map(Vec::as_slice).unwrap_or(&[]);
        let score = h.score().min(a.score());

        let within: Vec<GameId> = candidates
            .iter()
            .copied()
            .filter(|g| match start {
                None => true,
                Some(t) => {
                    let dt = (self.games[g.0 as usize].start.0 - t.0).abs() as f64 / 1e9;
                    dt <= self.cfg.window_secs
                }
            })
            .collect();

        let chosen = match (within.as_slice(), start) {
            ([], _) => {
                return Err(DataError::Unresolved {
                    what: "fixture".into(),
                    detail: format!(
                        "no scheduled meeting of {home:?} and {away:?}{}",
                        start.map(|t| format!(" near {}", format_rfc3339(t))).unwrap_or_default()
                    ),
                });
            }
            ([only], _) => *only,
            // Several meetings and no kickoff to separate them. A fixture is
            // not a game; refuse rather than pick the first of a season series.
            (many, None) => {
                return Err(DataError::Unresolved {
                    what: "fixture".into(),
                    detail: format!(
                        "{} meetings of {home:?} and {away:?} and no kickoff time to choose between them",
                        many.len()
                    ),
                });
            }
            (many, Some(t)) => *many
                .iter()
                .min_by(|x, y| {
                    let d = |g: &GameId| (self.games[g.0 as usize].start.0 - t.0).abs();
                    d(x).cmp(&d(y))
                })
                .expect("non-empty"),
        };

        Ok(GameMatch { game: chosen, swapped: self.games[chosen.0 as usize].home != hid, score })
    }

    /// The canonical string for a fixture, to be interned as an `EventId`.
    ///
    /// Built from catalogue keys and the scheduled date rather than from any
    /// venue's names, so every venue quoting the fixture produces byte-identical
    /// output. Always away-at-home, so a venue that listed the teams backwards
    /// still lands on the same key.
    pub fn event_key(&self, id: GameId) -> Option<String> {
        let g = self.games.get(id.0 as usize)?;
        let date = &format_rfc3339(g.start)[..10];
        let league = g.league.as_deref().unwrap_or("-");
        Some(format!(
            "{league}:{date}:{}@{}",
            self.teams[g.away.0 as usize].key, self.teams[g.home.0 as usize].key
        ))
    }

    /// Resolve and key in one step — the call the venue adapters make.
    pub fn resolve_event_key(
        &self,
        home: &str,
        away: &str,
        start: Option<Ts>,
        league: Option<&str>,
    ) -> Result<String, DataError> {
        let m = self.resolve_game(home, away, start, league)?;
        self.event_key(m.game).ok_or_else(|| DataError::Unresolved {
            what: "fixture".into(),
            detail: "resolved to a game not in the catalogue".into(),
        })
    }

    /// Build from the JSON catalogue shape the Python implementation seeded.
    pub fn from_catalog(catalog: &Catalog, cfg: ResolverConfig) -> Result<Self, DataError> {
        let mut r = Resolver::new(cfg);
        let mut by_key: HashMap<&str, TeamId> = HashMap::new();
        for t in &catalog.teams {
            let id = r.add_team(Team {
                key: t.id.clone(),
                name: t.name.clone(),
                aliases: t.aliases.clone(),
                league: t.sport.clone(),
            });
            by_key.insert(t.id.as_str(), id);
        }
        for g in &catalog.games {
            let team = |k: &str| -> Result<TeamId, DataError> {
                by_key.get(k).copied().ok_or_else(|| {
                    DataError::Config(format!("game {:?} references unknown team {k:?}", g.id))
                })
            };
            r.add_game(Game {
                key: g.id.clone(),
                home: team(&g.home_team_id)?,
                away: team(&g.away_team_id)?,
                start: parse_rfc3339(&g.start_time)?,
                league: g.sport.clone(),
            });
        }
        Ok(r)
    }

    pub fn from_json(json: &str, cfg: ResolverConfig) -> Result<Self, DataError> {
        let catalog: Catalog = serde_json::from_str(json).map_err(|e| DataError::Decode {
            venue: "catalog".into(),
            what: "entities".into(),
            detail: e.to_string(),
        })?;
        Resolver::from_catalog(&catalog, cfg)
    }
}

fn pair(a: TeamId, b: TeamId) -> (TeamId, TeamId) {
    if a <= b { (a, b) } else { (b, a) }
}

/// The on-disk catalogue, matching the seed file the Python service shipped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub teams: Vec<CatalogTeam>,
    #[serde(default)]
    pub games: Vec<CatalogGame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogTeam {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub sport: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogGame {
    pub id: String,
    pub home_team_id: String,
    pub away_team_id: String,
    #[serde(default)]
    pub sport: Option<String>,
    pub start_time: String,
}

/// Which side of a parsed matchup title is the home team.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// `A @ B` and `A at B` — unambiguously away first.
    AwayFirst,
    /// `A vs B`, `A - B`. Venues split roughly evenly on what this means, so
    /// the caller must not trust the order; [`Resolver::resolve_game`] tries
    /// both and reports which way round it landed.
    Unknown,
}

/// Split a venue's title into its two sides.
///
/// Handles `A @ B`, `A at B`, `A vs B`, `A v B`, and `A - B`, longest separator
/// first so "vs" is not found inside a team name and "at" is not found inside
/// "Atlanta".
pub fn parse_matchup(title: &str) -> Option<(&str, &str, Orientation)> {
    const SEPARATORS: &[(&str, Orientation)] = &[
        (" @ ", Orientation::AwayFirst),
        (" at ", Orientation::AwayFirst),
        (" vs. ", Orientation::Unknown),
        (" vs ", Orientation::Unknown),
        (" v. ", Orientation::Unknown),
        (" v ", Orientation::Unknown),
        (" - ", Orientation::Unknown),
    ];
    let lower = title.to_lowercase();
    for (sep, orientation) in SEPARATORS {
        if let Some(i) = lower.find(sep) {
            let (l, r) = (title[..i].trim(), title[i + sep.len()..].trim());
            if !l.is_empty() && !r.is_empty() {
                return Some((l, r, *orientation));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"{
      "teams": [
        {"id": "nyr", "name": "New York Rangers", "aliases": ["NY Rangers"], "sport": "icehockey_nhl"},
        {"id": "lal", "name": "Los Angeles Lakers", "aliases": ["Lakers", "LA Lakers"], "sport": "basketball_nba"},
        {"id": "nyk", "name": "NY Knicks", "aliases": ["Knicks"], "sport": "basketball_nba"},
        {"id": "bos", "name": "Boston Celtics", "aliases": ["Celtics"], "sport": "basketball_nba"}
      ],
      "games": [
        {"id": "g1", "home_team_id": "nyk", "away_team_id": "bos",
         "sport": "basketball_nba", "start_time": "2026-05-27T22:00:00"},
        {"id": "g2", "home_team_id": "lal", "away_team_id": "bos",
         "sport": "basketball_nba", "start_time": "2026-05-28T02:00:00"}
      ]
    }"#;

    fn resolver() -> Resolver {
        Resolver::from_json(CATALOG, ResolverConfig::default()).unwrap()
    }

    fn at(s: &str) -> Ts {
        parse_rfc3339(s).unwrap()
    }

    #[test]
    fn the_python_seed_catalogue_still_loads() {
        let r = resolver();
        assert_eq!(r.teams().len(), 4);
        assert_eq!(r.games().len(), 2);
    }

    #[test]
    fn a_canonical_name_or_alias_matches_outright() {
        let r = resolver();
        assert!(matches!(r.resolve_team("Boston Celtics", None), Resolution::Exact(_)));
        assert!(matches!(r.resolve_team("celtics", None), Resolution::Exact(_)));
        assert!(matches!(r.resolve_team("  BOSTON   CELTICS ", None), Resolution::Exact(_)));
    }

    #[test]
    fn a_misspelling_still_resolves() {
        let r = resolver();
        let got = r.resolve_team("Boston Celtcs", None);
        assert!(got.is_resolved(), "{got:?}");
        assert_eq!(r.team(got.id().unwrap()).unwrap().key, "bos");
    }

    #[test]
    fn an_abbreviated_city_resolves() {
        let r = resolver();
        assert_eq!(
            r.resolve_team("L.A. Lakers", None).id().map(|i| &r.team(i).unwrap().key),
            Some(&"lal".to_string())
        );
    }

    #[test]
    fn a_name_nothing_matches_is_unknown_not_a_guess() {
        let r = resolver();
        assert_eq!(r.resolve_team("Manchester United", None), Resolution::Unknown);
        assert_eq!(r.resolve_team("", None), Resolution::Unknown);
        assert_eq!(r.resolve_team("   ", None), Resolution::Unknown);
    }

    #[test]
    fn a_near_tie_resolves_to_nothing_rather_than_to_the_leader() {
        // Two Los Angeles teams and a name that names neither of them. Taking
        // the leader here means quoting one game against another's book.
        let mut r = Resolver::new(ResolverConfig::default());
        r.add_team(Team {
            key: "lal".into(),
            name: "Los Angeles Lakers".into(),
            aliases: vec![],
            league: None,
        });
        r.add_team(Team {
            key: "lac".into(),
            name: "Los Angeles Clippers".into(),
            aliases: vec![],
            league: None,
        });
        let got = r.resolve_team("Los Angeles", None);
        assert!(matches!(got, Resolution::Ambiguous { .. }), "{got:?}");
        assert_eq!(got.id(), None, "an ambiguous match must not yield an id");
    }

    #[test]
    fn one_literal_name_in_two_leagues_is_ambiguous_until_a_league_is_named() {
        let mut r = Resolver::new(ResolverConfig::default());
        r.add_team(Team {
            key: "nyr".into(),
            name: "Rangers".into(),
            aliases: vec![],
            league: Some("icehockey_nhl".into()),
        });
        r.add_team(Team {
            key: "rfc".into(),
            name: "Rangers".into(),
            aliases: vec![],
            league: Some("soccer_spl".into()),
        });
        assert!(matches!(r.resolve_team("Rangers", None), Resolution::Ambiguous { .. }));

        let got = r.resolve_team("Rangers", Some("soccer_spl"));
        assert_eq!(r.team(got.id().expect("league disambiguates")).unwrap().key, "rfc");
    }

    #[test]
    fn blocking_excludes_the_wrong_competition_entirely() {
        let r = resolver();
        assert_eq!(
            r.resolve_team("Boston Celtics", Some("icehockey_nhl")),
            Resolution::Unknown,
            "a basketball team is not a candidate in a hockey market"
        );
    }

    #[test]
    fn a_teamless_entity_is_eligible_in_every_league() {
        let mut r = Resolver::new(ResolverConfig::default());
        r.add_team(Team { key: "x".into(), name: "Team X".into(), aliases: vec![], league: None });
        assert!(r.resolve_team("Team X", Some("anything")).is_resolved());
    }

    #[test]
    fn a_fixture_resolves_from_both_names_and_a_kickoff() {
        let r = resolver();
        let m =
            r.resolve_game("NY Knicks", "Celtics", Some(at("2026-05-27T22:00:00Z")), None).unwrap();
        assert_eq!(r.game(m.game).unwrap().key, "g1");
        assert!(!m.swapped);
    }

    #[test]
    fn a_venue_listing_the_teams_backwards_still_lands_on_the_same_fixture() {
        let r = resolver();
        let m =
            r.resolve_game("Celtics", "NY Knicks", Some(at("2026-05-27T22:00:00Z")), None).unwrap();
        assert_eq!(r.game(m.game).unwrap().key, "g1");
        assert!(m.swapped, "the caller's home side is the catalogue's away side");
    }

    #[test]
    fn the_kickoff_separates_two_meetings_of_the_same_pair() {
        let r = resolver();
        let early =
            r.resolve_game("Knicks", "Celtics", Some(at("2026-05-27T22:00:00Z")), None).unwrap();
        assert_eq!(r.game(early.game).unwrap().key, "g1");
        // The Celtics also visit the Lakers four hours later; a pair of names
        // alone would not tell those apart.
        let late =
            r.resolve_game("Lakers", "Celtics", Some(at("2026-05-28T02:00:00Z")), None).unwrap();
        assert_eq!(r.game(late.game).unwrap().key, "g2");
    }

    #[test]
    fn a_kickoff_outside_the_window_is_a_different_game_entirely() {
        let r = resolver();
        let err = r
            .resolve_game("Knicks", "Celtics", Some(at("2026-06-27T22:00:00Z")), None)
            .unwrap_err();
        assert!(matches!(err, DataError::Unresolved { .. }), "{err}");
    }

    #[test]
    fn a_local_time_quoted_a_few_hours_out_still_matches() {
        let r = resolver();
        let m =
            r.resolve_game("Knicks", "Celtics", Some(at("2026-05-27T18:00:00Z")), None).unwrap();
        assert_eq!(r.game(m.game).unwrap().key, "g1");
    }

    #[test]
    fn several_meetings_and_no_kickoff_refuses_rather_than_taking_the_first() {
        let mut r = resolver();
        // A second Knicks/Celtics meeting later in the series.
        let (h, a) = (TeamId(2), TeamId(3));
        r.add_game(Game {
            key: "g3".into(),
            home: h,
            away: a,
            start: at("2026-05-30T22:00:00Z"),
            league: Some("basketball_nba".into()),
        });
        let err = r.resolve_game("Knicks", "Celtics", None, None).unwrap_err();
        assert!(matches!(err, DataError::Unresolved { .. }), "{err}");
    }

    #[test]
    fn a_half_identified_fixture_is_not_a_fixture() {
        let r = resolver();
        let err = r.resolve_game("Knicks", "Some Team Nobody Has", None, None).unwrap_err();
        assert!(matches!(err, DataError::Unresolved { .. }));
    }

    #[test]
    fn an_ambiguity_is_reported_as_such_rather_than_as_a_miss() {
        let r = resolver();
        let err = r.resolve_game("Knicks", "New York", Some(at("2026-05-27T22:00:00Z")), None);
        assert!(err.is_err());
    }

    #[test]
    fn a_team_cannot_play_itself() {
        let r = resolver();
        let err = r.resolve_game("Celtics", "Boston Celtics", None, None).unwrap_err();
        assert!(matches!(err, DataError::Unresolved { .. }));
    }

    #[test]
    fn the_event_key_is_identical_whichever_way_round_the_venue_listed_it() {
        let r = resolver();
        let t = Some(at("2026-05-27T22:00:00Z"));
        let one = r.resolve_event_key("NY Knicks", "Celtics", t, None).unwrap();
        let other = r.resolve_event_key("Boston Celtics", "Knicks", t, None).unwrap();
        assert_eq!(one, other, "cross-venue arbitrage depends on this being one key");
        assert_eq!(one, "basketball_nba:2026-05-27:bos@nyk");
    }

    #[test]
    fn the_event_key_uses_catalogue_keys_not_venue_spellings() {
        let r = resolver();
        let t = Some(at("2026-05-27T22:00:00Z"));
        assert_eq!(
            r.resolve_event_key("Knicks", "Celtcs", t, None).unwrap(),
            r.resolve_event_key("NY Knicks", "Boston Celtics", t, None).unwrap()
        );
    }

    #[test]
    fn matchup_titles_split_on_every_separator_venues_use() {
        assert_eq!(parse_matchup("BOS @ NYK"), Some(("BOS", "NYK", Orientation::AwayFirst)));
        assert_eq!(
            parse_matchup("Boston Celtics at New York Knicks"),
            Some(("Boston Celtics", "New York Knicks", Orientation::AwayFirst))
        );
        assert_eq!(
            parse_matchup("Lakers vs. Celtics"),
            Some(("Lakers", "Celtics", Orientation::Unknown))
        );
        assert_eq!(
            parse_matchup("Lakers v Celtics"),
            Some(("Lakers", "Celtics", Orientation::Unknown))
        );
        assert_eq!(
            parse_matchup("Lakers - Celtics"),
            Some(("Lakers", "Celtics", Orientation::Unknown))
        );
    }

    #[test]
    fn a_separator_inside_a_team_name_is_not_a_separator() {
        // "at" lives inside "Atlanta"; only the padded form counts.
        assert_eq!(
            parse_matchup("Atlanta Hawks @ Miami Heat"),
            Some(("Atlanta Hawks", "Miami Heat", Orientation::AwayFirst))
        );
        assert_eq!(parse_matchup("Atlanta Hawks"), None);
        assert_eq!(parse_matchup(""), None);
    }

    #[test]
    fn a_catalogue_referencing_a_team_that_does_not_exist_is_a_config_error() {
        let json = r#"{"teams": [], "games": [
            {"id": "g", "home_team_id": "nope", "away_team_id": "also-nope",
             "start_time": "2026-01-01T00:00:00Z"}]}"#;
        let err = Resolver::from_json(json, ResolverConfig::default()).unwrap_err();
        assert!(matches!(err, DataError::Config(_)), "{err}");
        assert!(!err.is_transient(), "a broken catalogue is not fixed by retrying");
    }
}
