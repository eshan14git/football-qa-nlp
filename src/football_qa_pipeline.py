"""Shared data pipeline for the Football NLP Question Answering project.

This module contains the dataset preparation shared by the ML and DL
experiments.  It converts the three raw football CSV files into the validated
five-intent QA dataset, creates match-level splits, and exports CSV/JSONL files.

Typical Colab use::

    from football_qa_pipeline import run_pipeline

    artifacts = run_pipeline(
        data_directory="/content/drive/MyDrive/NLP/data",
        export_directory="/content/drive/MyDrive/NLP/exports",
    )

For normal model experimentation, do not rebuild the dataset. Load the already
prepared split files instead::

    from football_qa_pipeline import load_prepared_splits

    train_df, validation_df, test_df = load_prepared_splits(
        "/content/drive/MyDrive/NLP/exports/splits"
    )
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


RANDOM_SEED = 42
MATCH_KEY = ["date_parsed", "home_team_key", "away_team_key"]
EXPECTED_INTENTS = {
    "match_winner",
    "match_score",
    "match_scorers",
    "player_match_goal_count",
    "player_match_scoring_minutes",
}
PREFERRED_COLUMN_ORDER = [
    "question_id",
    "match_id",
    "intent",
    "question",
    "answer",
    "source_dataset",
    "scorer",
    "team",
    "player_goal_count",
    "goal_count",
    "scoring_minutes",
    "date_standardized",
    "home_team_standardized",
    "away_team_standardized",
    "home_score",
    "away_score",
    "tournament",
]

V2_METADATA_COLUMNS = [
    "semantic_question_id",
    "template_id",
    "template_family",
    "template_seen_in_training",
    "evaluation_track",
    "question_variant",
    "date_format",
    "team_order",
]


def _templates(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    return [
        {"template_id": template_id, "template_family": family, "text": text}
        for template_id, family, text in items
    ]


# Reviewed rather than machine-generated so every question remains reproducible
# and semantically equivalent to its Version 1 source row.
SEEN_TEMPLATES = {
    "match_winner": _templates(
        ("WIN_SEEN_01", "direct", "Who won, or was it a draw, when {home} played {away} on {date}?"),
        ("WIN_SEEN_02", "victor", "Did either team win when {home} faced {away} on {date}?"),
        ("WIN_SEEN_03", "fixture", "Was there a winner in the {home} versus {away} fixture on {date}, and if so, who?"),
        ("WIN_SEEN_04", "choice", "Did {home} win, did {away} win, or was their game on {date} drawn?"),
        ("WIN_SEEN_05", "side", "Which side won on {date}, {home} or {away}, or did they draw?"),
        ("WIN_SEEN_06", "outcome", "What was the outcome when {home} played {away} on {date}?"),
        ("WIN_SEEN_07", "result", "How did the match between {home} and {away} finish on {date}?"),
        ("WIN_SEEN_08", "secured", "When {home} met {away} on {date}, which team won, or did neither side win?"),
    ),
    "match_score": _templates(
        ("SCORE_SEEN_01", "direct", "What was the score between {home} and {away} on {date}?"),
        ("SCORE_SEEN_02", "final_score", "What was the final score when {home} faced {away} on {date}?"),
        ("SCORE_SEEN_03", "fixture", "How did the {home} versus {away} fixture end on {date}?"),
        ("SCORE_SEEN_04", "scoreline", "Give the scoreline for {home} against {away} on {date}."),
        ("SCORE_SEEN_05", "result", "What result was recorded when {home} played {away} on {date}?"),
        ("SCORE_SEEN_06", "ended", "With what score did {home}'s match against {away} end on {date}?"),
        ("SCORE_SEEN_07", "report", "Report the final score of {home} versus {away} on {date}."),
        ("SCORE_SEEN_08", "goals", "How many goals did each side score when {home} met {away} on {date}?"),
    ),
    "match_scorers": _templates(
        ("SCORERS_SEEN_01", "direct", "Who scored in the match between {home} and {away} on {date}?"),
        ("SCORERS_SEEN_02", "net", "Which players found the net when {home} faced {away} on {date}?"),
        ("SCORERS_SEEN_03", "goals", "Who scored the goals in the {home} versus {away} fixture on {date}?"),
        ("SCORERS_SEEN_04", "scoresheet", "Who got on the scoresheet for {home} and {away} on {date}?"),
        ("SCORERS_SEEN_05", "names", "Name the goalscorers from {home}'s match against {away} on {date}."),
        ("SCORERS_SEEN_06", "recorded", "Which players recorded goals when {home} played {away} on {date}?"),
        ("SCORERS_SEEN_07", "responsible", "Who was responsible for the goals between {home} and {away} on {date}?"),
        ("SCORERS_SEEN_08", "list", "List every scorer in the {home}-{away} match played on {date}."),
    ),
    "player_match_goal_count": _templates(
        ("PGCOUNT_SEEN_01", "direct", "How many goals did {player} score in the match between {home} and {away} on {date}?"),
        ("PGCOUNT_SEEN_02", "count", "What was {player}'s goal count when {home} faced {away} on {date}?"),
        ("PGCOUNT_SEEN_03", "times", "How many times did {player} score in the {home} versus {away} fixture on {date}?"),
        ("PGCOUNT_SEEN_04", "contribution", "How many goals did {player} contribute when {home} played {away} on {date}?"),
        ("PGCOUNT_SEEN_05", "scoresheet", "How many goals did {player} record in {home} against {away} on {date}?"),
        ("PGCOUNT_SEEN_06", "total", "What total did {player} score during the {home}-{away} match on {date}?"),
        ("PGCOUNT_SEEN_07", "record", "State {player}'s number of goals when {home} met {away} on {date}."),
        ("PGCOUNT_SEEN_08", "net", "How many goals did {player} put in the net in {home} versus {away} on {date}?"),
    ),
    "player_match_scoring_minutes": _templates(
        ("PGMINUTE_SEEN_01", "direct", "When did {player} score in the match between {home} and {away} on {date}?"),
        ("PGMINUTE_SEEN_02", "minutes", "In which minutes did {player} score when {home} faced {away} on {date}?"),
        ("PGMINUTE_SEEN_03", "timing", "What were the scoring times for {player} in {home} versus {away} on {date}?"),
        ("PGMINUTE_SEEN_04", "goals", "At what points did {player}'s goals occur when {home} played {away} on {date}?"),
        ("PGMINUTE_SEEN_05", "recorded", "Which goal minutes were recorded for {player} in the {home}-{away} match on {date}?"),
        ("PGMINUTE_SEEN_06", "scoreboard", "At what times did {player} score in the match between {home} and {away} on {date}?"),
        ("PGMINUTE_SEEN_07", "list", "List the minutes in which {player} scored during {home} versus {away} on {date}."),
        ("PGMINUTE_SEEN_08", "timed", "How were {player}'s goals timed when {home} met {away} on {date}?"),
    ),
}

UNSEEN_TEMPLATES = {
    "match_winner": _templates(
        ("WIN_UNSEEN_01", "honours", "When {home} and {away} contested their {date} match, who took the honours, or was it even?"),
        ("WIN_UNSEEN_02", "prevailed", "Identify the side that prevailed in {home} against {away} on {date}, or state that they drew."),
        ("WIN_UNSEEN_03", "victory_or_draw", "Was it {home}, {away}, or neither that won on {date}?"),
        ("WIN_UNSEEN_04", "points", "Who came out on top when {home} crossed paths with {away} on {date}, or did the match end level?"),
    ),
    "match_score": _templates(
        ("SCORE_UNSEEN_01", "numbers", "What numbers appeared on the final scoreboard for {home} and {away} on {date}?"),
        ("SCORE_UNSEEN_02", "tally", "Provide the full-time tally from {home} against {away}, played on {date}."),
        ("SCORE_UNSEEN_03", "goals_each", "How many goals did {home} and {away} each record on {date}?"),
        ("SCORE_UNSEEN_04", "read", "How did the scoreboard read after {home} met {away} on {date}?"),
    ),
    "match_scorers": _templates(
        ("SCORERS_UNSEEN_01", "goal_authors", "Identify the authors of every goal in {home} against {away} on {date}."),
        ("SCORERS_UNSEEN_02", "scorer_names", "Which players scored when {home} met {away} on {date}?"),
        ("SCORERS_UNSEEN_03", "converted", "Which players converted chances into goals in {home} versus {away} on {date}?"),
        ("SCORERS_UNSEEN_04", "accounted", "Who accounted for the scoring in the {date} meeting of {home} and {away}?"),
    ),
    "player_match_goal_count": _templates(
        ("PGCOUNT_UNSEEN_01", "personal_tally", "What personal tally did {player} finish with when {home} met {away} on {date}?"),
        ("PGCOUNT_UNSEEN_02", "credited", "How many of the goals in {home} versus {away} on {date} were credited to {player}?"),
        ("PGCOUNT_UNSEEN_03", "entries", "How many scoring entries belonged to {player} in the {date} match between {home} and {away}?"),
        ("PGCOUNT_UNSEEN_04", "bagged", "How many did {player} bag when {home} took on {away} on {date}?"),
    ),
    "player_match_scoring_minutes": _templates(
        ("PGMINUTE_UNSEEN_01", "goal_times", "Give the scoring minute for every {player} goal in {home} against {away} on {date}."),
        ("PGMINUTE_UNSEEN_02", "clock", "At which readings of the match clock did {player} score when {home} met {away} on {date}?"),
        ("PGMINUTE_UNSEEN_03", "occurred", "When, during {home} versus {away} on {date}, did {player}'s goals occur?"),
        ("PGMINUTE_UNSEEN_04", "marked", "Which minutes were marked by goals from {player} in the {date} meeting of {home} and {away}?"),
    ),
}


def parse_football_dates(date_series: pd.Series) -> pd.Series:
    """Parse ISO dates and US-style M/D/YYYY dates without ambiguity."""
    date_text = date_series.astype(str).str.strip()
    iso_mask = date_text.str.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}")
    parsed = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(
        date_text.loc[iso_mask], format="%Y-%m-%d", errors="coerce"
    )
    parsed.loc[~iso_mask] = pd.to_datetime(
        date_text.loc[~iso_mask], format="%m/%d/%Y", errors="coerce"
    )
    return parsed


def parse_scoring_minute(value: object) -> pd.Series:
    """Split a football minute such as ``90+3`` into useful components."""
    if pd.isna(value):
        return pd.Series(
            {
                "minute_original": pd.NA,
                "base_minute": pd.NA,
                "added_time": pd.NA,
                "minute_total": pd.NA,
            }
        )

    minute_text = str(value).strip()
    try:
        if "+" in minute_text:
            base_text, added_text = minute_text.split("+", maxsplit=1)
            base_minute = int(float(base_text))
            added_time = int(float(added_text))
            minute_total = base_minute + added_time
        else:
            base_minute = int(float(minute_text))
            added_time = 0
            minute_total = base_minute
    except ValueError:
        base_minute = added_time = minute_total = pd.NA

    return pd.Series(
        {
            "minute_original": minute_text,
            "base_minute": base_minute,
            "added_time": added_time,
            "minute_total": minute_total,
        }
    )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize common CSV boolean representations, defaulting missing to False."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def clean_minute_text(value: object) -> str:
    minute_text = str(value).strip()
    return minute_text[:-2] if minute_text.endswith(".0") else minute_text


def minute_sort_value(value: object) -> float:
    minute_text = clean_minute_text(value)
    if "+" in minute_text:
        base, added = minute_text.split("+", maxsplit=1)
        return int(base) + int(added) / 100
    return float(int(minute_text))


def join_naturally(values: pd.Series | list[str]) -> str:
    items = list(values)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def join_scoring_minutes(values: pd.Series) -> str:
    return join_naturally([f"{value}'" for value in values])


def load_raw_data(data_directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``results.csv``, ``goalscorers.csv``, and ``shootouts.csv``."""
    data_directory = Path(data_directory)
    paths = {
        name: data_directory / f"{name}.csv"
        for name in ("results", "goalscorers", "shootouts")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw dataset file(s): " + ", ".join(missing))
    return tuple(pd.read_csv(paths[name]) for name in paths)  # type: ignore[return-value]


def _standardize_link_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = dataframe.copy()
    prepared["date_parsed"] = parse_football_dates(prepared["date"])
    for side in ("home", "away"):
        prepared[f"{side}_team_key"] = (
            prepared[f"{side}_team"].astype(str).str.strip().str.casefold()
        )
    return prepared


def prepare_safe_sources(
    results_df: pd.DataFrame,
    goalscorers_df: pd.DataFrame,
    shootouts_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize linkage fields and exclude ambiguous result match keys."""
    results = _standardize_link_fields(results_df)
    goalscorers = _standardize_link_fields(goalscorers_df)
    shootouts = _standardize_link_fields(shootouts_df)

    result_key_counts = (
        results.groupby(MATCH_KEY, dropna=False).size().reset_index(name="count")
    )
    safe_keys = result_key_counts.loc[result_key_counts["count"].eq(1), MATCH_KEY]

    safe_results = results.merge(safe_keys, on=MATCH_KEY, how="inner").copy()
    safe_goalscorers = goalscorers.merge(
        safe_keys.assign(safe_result_key=True), on=MATCH_KEY, how="inner"
    ).copy()
    safe_shootouts = shootouts.merge(
        safe_keys.assign(safe_result_key=True), on=MATCH_KEY, how="inner"
    ).copy()

    for dataframe in (safe_results, safe_goalscorers, safe_shootouts):
        dataframe["date_standardized"] = dataframe["date_parsed"].dt.strftime(
            "%Y-%m-%d"
        )
        dataframe["home_team_standardized"] = dataframe["home_team"].astype(str).str.strip()
        dataframe["away_team_standardized"] = dataframe["away_team"].astype(str).str.strip()

    minute_fields = safe_goalscorers["minute"].apply(parse_scoring_minute)
    safe_goalscorers = pd.concat(
        [safe_goalscorers.reset_index(drop=True), minute_fields.reset_index(drop=True)],
        axis=1,
    )
    for column in ("base_minute", "added_time", "minute_total"):
        safe_goalscorers[column] = safe_goalscorers[column].astype("Int64")

    return safe_results, safe_goalscorers, safe_shootouts


def _result_question_source(final_results_df: pd.DataFrame) -> pd.DataFrame:
    source = final_results_df[
        [
            "date_standardized",
            "home_team_standardized",
            "away_team_standardized",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ]
    ].copy()
    source[["home_score", "away_score"]] = source[
        ["home_score", "away_score"]
    ].astype(int)
    source["match_id"] = [f"MATCH_{number:05d}" for number in range(1, len(source) + 1)]
    return source


def _winner_answer(row: pd.Series) -> str:
    home, away = row["home_team_standardized"], row["away_team_standardized"]
    home_score, away_score = int(row["home_score"]), int(row["away_score"])
    date = row["date_standardized"]
    if home_score > away_score:
        return f"{home} won against {away} {home_score}-{away_score} on {date}."
    if away_score > home_score:
        return f"{away} won against {home} {away_score}-{home_score} on {date}."
    return f"The match between {home} and {away} ended in a {home_score}-{away_score} draw on {date}."


def _base_result_columns() -> list[str]:
    return [
        "question_id", "match_id", "intent", "question", "answer",
        "date_standardized", "home_team_standardized", "away_team_standardized",
        "home_score", "away_score", "tournament",
    ]


def build_result_questions(source: pd.DataFrame) -> pd.DataFrame:
    """Create match-winner and match-score QA rows."""
    winner = source.copy()
    winner["question_id"] = [f"WIN_{number:05d}" for number in range(1, len(winner) + 1)]
    winner["intent"] = "match_winner"
    winner["question"] = (
        "Who won the match between " + winner["home_team_standardized"] + " and "
        + winner["away_team_standardized"] + " on " + winner["date_standardized"] + "?"
    )
    winner["answer"] = winner.apply(_winner_answer, axis=1)

    score = source.copy()
    score["question_id"] = [f"SCORE_{number:05d}" for number in range(1, len(score) + 1)]
    score["intent"] = "match_score"
    score["question"] = (
        "What was the score between " + score["home_team_standardized"] + " and "
        + score["away_team_standardized"] + " on " + score["date_standardized"] + "?"
    )
    score["answer"] = (
        score["home_team_standardized"] + " " + score["home_score"].astype(str)
        + "-" + score["away_score"].astype(str) + " "
        + score["away_team_standardized"] + "."
    )
    columns = _base_result_columns()
    return pd.concat([winner[columns], score[columns]], ignore_index=True)


def _reliable_scorer_events(
    goalscorers: pd.DataFrame,
    results: pd.DataFrame,
    result_source: pd.DataFrame,
) -> pd.DataFrame:
    lookup = results[MATCH_KEY].copy()
    lookup["match_id"] = result_source["match_id"].values
    events = goalscorers.merge(lookup, on=MATCH_KEY, how="left", validate="many_to_one")
    if events["match_id"].isna().any():
        raise ValueError("Some safely linked goalscorer rows have no match_id.")

    coverage = (
        events.groupby("match_id")
        .agg(
            recorded_goal_events=("match_id", "size"),
            missing_scorer_events=("scorer", lambda values: values.isna().sum()),
        )
        .reset_index()
        .merge(
            result_source[["match_id", "home_score", "away_score"]],
            on="match_id", how="left", validate="one_to_one",
        )
    )
    coverage["result_total_goals"] = coverage["home_score"] + coverage["away_score"]
    reliable_ids = set(
        coverage.loc[
            coverage["recorded_goal_events"].eq(coverage["result_total_goals"])
            & coverage["missing_scorer_events"].eq(0),
            "match_id",
        ]
    )
    return events.loc[events["match_id"].isin(reliable_ids)].copy()


def _scorer_summary(row: pd.Series) -> str:
    scorer, team = row["scorer"], row["team"]
    goal_count, penalty_count = int(row["goal_count"]), int(row["penalty_count"])
    if row["own_goal"]:
        if goal_count == 1:
            return f"{scorer} scored an own goal credited to {team}"
        return f"{scorer} scored {goal_count} own goals credited to {team}"
    word = "goal" if goal_count == 1 else "goals"
    summary = f"{scorer} scored {goal_count} {word} for {team}"
    if penalty_count == 1:
        summary += ", including 1 penalty"
    elif penalty_count > 1:
        summary += f", including {penalty_count} penalties"
    return summary


def build_scorer_questions(
    reliable_events: pd.DataFrame, result_source: pd.DataFrame
) -> pd.DataFrame:
    """Create one aggregated, naturally worded scorer-list question per match."""
    events = reliable_events[
        ["match_id", "scorer", "team", "own_goal", "penalty", "minute_total"]
    ].copy()
    events["own_goal"] = normalize_boolean(events["own_goal"])
    events["penalty"] = normalize_boolean(events["penalty"])
    events["source_order"] = range(len(events))
    events = events.sort_values(
        ["match_id", "minute_total", "source_order"],
        kind="stable", na_position="last",
    ).reset_index(drop=True)
    events["event_order"] = events.groupby("match_id").cumcount()

    aggregated = (
        events.groupby(
            ["match_id", "scorer", "team", "own_goal"],
            as_index=False, sort=False, dropna=False,
        )
        .agg(
            goal_count=("scorer", "size"),
            penalty_count=("penalty", "sum"),
            first_event_order=("event_order", "min"),
        )
        .sort_values(["match_id", "first_event_order"], kind="stable")
        .reset_index(drop=True)
    )
    aggregated["scorer_summary"] = aggregated.apply(_scorer_summary, axis=1)
    answers = (
        aggregated.groupby("match_id", sort=False)["scorer_summary"]
        .apply(lambda values: join_naturally(values) + ".")
        .rename("answer").reset_index()
    )
    qa = answers.merge(
        result_source[
            ["match_id", "date_standardized", "home_team_standardized",
             "away_team_standardized", "home_score", "away_score", "tournament"]
        ],
        on="match_id", how="left", validate="one_to_one",
    )
    qa["question_id"] = [f"SCORERS_{number:05d}" for number in range(1, len(qa) + 1)]
    qa["intent"] = "match_scorers"
    qa["question"] = (
        "Who scored in the match between " + qa["home_team_standardized"] + " and "
        + qa["away_team_standardized"] + " on " + qa["date_standardized"] + "?"
    )
    return qa[_base_result_columns()]


def build_player_goal_count_questions(
    reliable_events: pd.DataFrame, result_source: pd.DataFrame
) -> pd.DataFrame:
    """Create player-within-match goal-count QA rows, excluding own goals."""
    events = reliable_events.loc[~normalize_boolean(reliable_events["own_goal"])].copy()
    source = (
        events.groupby(["match_id", "scorer", "team"], sort=False, dropna=False)
        .agg(player_goal_count=("scorer", "size"))
        .reset_index()
        .merge(
            result_source[["match_id", "date_standardized", "home_team_standardized",
                           "away_team_standardized", "home_score", "away_score", "tournament"]],
            on="match_id", how="left", validate="many_to_one",
        )
    )
    qa = source.copy()
    qa["question_id"] = [f"PGCOUNT_{number:05d}" for number in range(1, len(qa) + 1)]
    qa["intent"] = "player_match_goal_count"
    qa["question"] = (
        "How many goals did " + qa["scorer"] + " score in the match between "
        + qa["home_team_standardized"] + " and " + qa["away_team_standardized"]
        + " on " + qa["date_standardized"] + "?"
    )
    qa["answer"] = qa.apply(
        lambda row: (
            f"{row['scorer']} scored {int(row['player_goal_count'])} "
            f"{'goal' if int(row['player_goal_count']) == 1 else 'goals'} "
            f"for {row['team']} in the match."
        ),
        axis=1,
    )
    return qa[
        ["question_id", "match_id", "intent", "question", "answer", "scorer",
         "team", "player_goal_count", "date_standardized", "home_team_standardized",
         "away_team_standardized", "home_score", "away_score", "tournament"]
    ]


def build_player_minute_questions(
    reliable_events: pd.DataFrame, result_source: pd.DataFrame
) -> pd.DataFrame:
    """Create scoring-minute QA rows only when all of a player's minutes are known."""
    player_events = reliable_events.loc[~normalize_boolean(reliable_events["own_goal"])].copy()
    completeness = (
        player_events.groupby(["match_id", "scorer", "team"], sort=False, dropna=False)
        .agg(missing_minutes=("minute_original", lambda values: values.isna().sum()))
        .reset_index()
    )
    reliable_groups = completeness.loc[
        completeness["missing_minutes"].eq(0), ["match_id", "scorer", "team"]
    ]
    events = player_events.merge(
        reliable_groups,
        on=["match_id", "scorer", "team"], how="inner", validate="many_to_one",
    )
    events["minute_text"] = events["minute_original"].apply(clean_minute_text)
    events["minute_sort_value"] = events["minute_original"].apply(minute_sort_value)
    source = (
        events.sort_values(["match_id", "scorer", "team", "minute_sort_value"])
        .groupby(["match_id", "scorer", "team"], sort=False, dropna=False)
        .agg(scoring_minutes=("minute_text", join_scoring_minutes), goal_count=("scorer", "size"))
        .reset_index()
        .merge(
            result_source[["match_id", "date_standardized", "home_team_standardized",
                           "away_team_standardized", "home_score", "away_score", "tournament"]],
            on="match_id", how="left", validate="many_to_one",
        )
    )
    qa = source.copy()
    qa["question_id"] = [f"PGMINUTE_{number:05d}" for number in range(1, len(qa) + 1)]
    qa["intent"] = "player_match_scoring_minutes"
    qa["question"] = (
        "When did " + qa["scorer"] + " score in the match between "
        + qa["home_team_standardized"] + " and " + qa["away_team_standardized"]
        + " on " + qa["date_standardized"] + "?"
    )
    qa["answer"] = qa.apply(
        lambda row: f"{row['scorer']} scored at {row['scoring_minutes']}.", axis=1
    )
    return qa[
        ["question_id", "match_id", "intent", "question", "answer", "scorer", "team",
         "goal_count", "scoring_minutes", "date_standardized", "home_team_standardized",
         "away_team_standardized", "home_score", "away_score", "tournament"]
    ]


def build_qa_dataset(
    results_df: pd.DataFrame,
    goalscorers_df: pd.DataFrame,
    shootouts_df: pd.DataFrame,
    *,
    enforce_reference_counts: bool = False,
) -> pd.DataFrame:
    """Build the complete five-intent QA dataset from the three raw tables.

    Set ``enforce_reference_counts=True`` when using the exact verified dataset
    snapshot from the notebook. This checks the known 194,159-row result.
    """
    results, goalscorers, _shootouts = prepare_safe_sources(
        results_df, goalscorers_df, shootouts_df
    )
    result_source = _result_question_source(results)
    reliable_events = _reliable_scorer_events(goalscorers, results, result_source)

    qa_sources = {
        "result_based": build_result_questions(result_source),
        "scorer_list": build_scorer_questions(reliable_events, result_source),
        "player_goal_count": build_player_goal_count_questions(reliable_events, result_source),
        "player_goal_minute": build_player_minute_questions(reliable_events, result_source),
    }
    parts = []
    for source_name, source_df in qa_sources.items():
        prepared = source_df.copy()
        prepared["source_dataset"] = source_name
        parts.append(prepared)
    final_qa_df = pd.concat(parts, ignore_index=True, sort=False)[PREFERRED_COLUMN_ORDER]
    validate_qa_dataset(final_qa_df)

    if enforce_reference_counts:
        expected_counts = {
            "match_winner": 49_481,
            "match_score": 49_481,
            "match_scorers": 15_508,
            "player_match_goal_count": 39_931,
            "player_match_scoring_minutes": 39_758,
        }
        actual_counts = final_qa_df["intent"].value_counts().to_dict()
        if len(final_qa_df) != 194_159 or actual_counts != expected_counts:
            raise ValueError(
                "Pipeline output does not match the verified reference snapshot. "
                f"Rows={len(final_qa_df):,}; intent counts={actual_counts}"
            )
    return final_qa_df


def validate_qa_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate structure, identifiers, text fields, and intent membership."""
    required = {"question_id", "match_id", "intent", "question", "answer"}
    missing_columns = required.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required QA columns: {sorted(missing_columns)}")

    checks = pd.DataFrame(
        [
            {"check": "duplicate_question_ids", "value": int(dataframe["question_id"].duplicated().sum())},
            {"check": "duplicate_questions", "value": int(dataframe["question"].duplicated().sum())},
            {"check": "missing_required_values", "value": int(dataframe[list(required)].isna().sum().sum())},
            {"check": "blank_questions", "value": int(dataframe["question"].astype("string").str.strip().eq("").sum())},
            {"check": "blank_answers", "value": int(dataframe["answer"].astype("string").str.strip().eq("").sum())},
        ]
    )
    if not checks["value"].eq(0).all():
        raise ValueError("QA validation failed:\n" + checks.to_string(index=False))
    intents = set(dataframe["intent"].unique())
    if intents != EXPECTED_INTENTS:
        raise ValueError(f"Unexpected intents. Expected {EXPECTED_INTENTS}; received {intents}")
    return checks


def create_match_level_splits(
    final_qa_df: pd.DataFrame,
    *,
    train_size: float = 0.80,
    random_state: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Create leakage-safe train/validation/test splits grouped by match_id."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    first = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_indices, temporary_indices = next(
        first.split(final_qa_df, groups=final_qa_df["match_id"])
    )
    train = final_qa_df.iloc[train_indices].copy().reset_index(drop=True)
    temporary = final_qa_df.iloc[temporary_indices].copy().reset_index(drop=True)

    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    validation_indices, test_indices = next(
        second.split(temporary, groups=temporary["match_id"])
    )
    splits = {
        "train": train,
        "validation": temporary.iloc[validation_indices].copy().reset_index(drop=True),
        "test": temporary.iloc[test_indices].copy().reset_index(drop=True),
    }
    validate_splits(final_qa_df, splits)
    return splits


def validate_splits(
    master_df: pd.DataFrame, splits: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Confirm full row preservation, unique IDs, all intents, and no match leakage."""
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Splits must contain train, validation, and test.")
    match_sets = {name: set(df["match_id"]) for name, df in splits.items()}
    overlaps = {
        "train_validation": len(match_sets["train"] & match_sets["validation"]),
        "train_test": len(match_sets["train"] & match_sets["test"]),
        "validation_test": len(match_sets["validation"] & match_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Match leakage detected: {overlaps}")

    combined_ids = pd.concat([df["question_id"] for df in splits.values()], ignore_index=True)
    if len(combined_ids) != len(master_df) or not combined_ids.is_unique:
        raise ValueError("Split rows do not preserve the unique master question IDs.")
    if set(combined_ids) != set(master_df["question_id"]):
        raise ValueError("Split question IDs differ from the master dataset.")
    for name, dataframe in splits.items():
        if set(dataframe["intent"].unique()) != EXPECTED_INTENTS:
            raise ValueError(f"The {name} split does not contain all five intents.")

    return pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(dataframe),
                "row_percentage": round(len(dataframe) / len(master_df) * 100, 2),
                "unique_matches": dataframe["match_id"].nunique(),
            }
            for name, dataframe in splits.items()
        ]
    )


def export_prepared_data(
    final_qa_df: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    export_directory: str | Path,
) -> dict[str, Path]:
    """Export master/split CSV and JSONL files plus match assignments."""
    export_directory = Path(export_directory)
    split_directory = export_directory / "splits"
    split_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for extension in ("csv", "jsonl"):
        path = export_directory / f"football_qa_master.{extension}"
        _write_dataframe(final_qa_df, path)
        paths[f"master_{extension}"] = path

    assignment_parts = []
    for split_name, split_df in splits.items():
        for extension in ("csv", "jsonl"):
            path = split_directory / f"football_qa_{split_name}.{extension}"
            _write_dataframe(split_df, path)
            paths[f"{split_name}_{extension}"] = path
        part = pd.DataFrame({"match_id": sorted(split_df["match_id"].unique())})
        part["split"] = split_name
        assignment_parts.append(part)

    assignments = (
        pd.concat(assignment_parts, ignore_index=True)
        .sort_values("match_id").reset_index(drop=True)
    )
    assignment_path = split_directory / "football_qa_match_split_assignments.csv"
    assignments.to_csv(assignment_path, index=False, encoding="utf-8")
    paths["match_split_assignments"] = assignment_path
    validate_exported_files(final_qa_df, splits, paths)
    return paths


def _write_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        dataframe.to_csv(path, index=False, encoding="utf-8")
    elif path.suffix == ".jsonl":
        dataframe.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported export extension: {path.suffix}")


def validate_exported_files(
    master_df: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    paths: Mapping[str, Path],
) -> None:
    """Reload exported files and validate their shape and identities."""
    for prefix, expected_df in {"master": master_df, **splits}.items():
        csv_df = pd.read_csv(paths[f"{prefix}_csv"], low_memory=False)
        jsonl_df = pd.read_json(paths[f"{prefix}_jsonl"], lines=True)
        for exported in (csv_df, jsonl_df):
            if len(exported) != len(expected_df):
                raise ValueError(f"Exported {prefix} row count is incorrect.")
            if list(exported.columns) != list(master_df.columns):
                raise ValueError(f"Exported {prefix} columns are incorrect.")
            if not exported["question_id"].is_unique:
                raise ValueError(f"Exported {prefix} question IDs are not unique.")
            if not exported[["question", "answer"]].notna().all().all():
                raise ValueError(f"Exported {prefix} contains missing QA text.")


def load_prepared_splits(
    split_directory: str | Path,
    *,
    file_format: str = "csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prepared train, validation, and test files for ML or DL notebooks."""
    if file_format not in {"csv", "jsonl"}:
        raise ValueError("file_format must be 'csv' or 'jsonl'.")
    split_directory = Path(split_directory)
    loaded = []
    for name in ("train", "validation", "test"):
        path = split_directory / f"football_qa_{name}.{file_format}"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared split not found: {path}")
        dataframe = (
            pd.read_csv(path, low_memory=False)
            if file_format == "csv"
            else pd.read_json(path, lines=True)
        )
        loaded.append(dataframe)
    return loaded[0], loaded[1], loaded[2]


def _stable_number(*parts: object, random_state: int = RANDOM_SEED) -> int:
    payload = "|".join(str(part) for part in (*parts, random_state)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _format_question_date(date_text: str, format_name: str) -> str:
    date = pd.Timestamp(date_text)
    if format_name == "iso":
        return date.strftime("%Y-%m-%d")
    if format_name == "day_month_year":
        return f"{date.day} {date.strftime('%B')} {date.year}"
    if format_name == "month_day_year":
        return f"{date.strftime('%B')} {date.day}, {date.year}"
    if format_name == "day_short_month_year":
        return f"{date.day} {date.strftime('%b')} {date.year}"
    raise ValueError(f"Unsupported date format: {format_name}")


def _render_v2_row(
    row: Mapping[str, object],
    *,
    templates: Mapping[str, list[dict[str, str]]],
    evaluation_track: str,
    seen_in_training: bool,
    random_state: int,
) -> dict[str, object]:
    semantic_id = str(row["question_id"])
    intent = str(row["intent"])
    candidates = templates[intent]
    template = candidates[
        _stable_number(semantic_id, evaluation_track, "template", random_state=random_state)
        % len(candidates)
    ]
    date_formats = ("iso", "day_month_year", "month_day_year", "day_short_month_year")
    date_format = date_formats[
        _stable_number(semantic_id, evaluation_track, "date", random_state=random_state)
        % len(date_formats)
    ]
    reverse = bool(
        _stable_number(semantic_id, evaluation_track, "order", random_state=random_state)
        % 2
    )
    original_home = str(row["home_team_standardized"])
    original_away = str(row["away_team_standardized"])
    home, away = (
        (original_away, original_home) if reverse else (original_home, original_away)
    )
    player = "" if pd.isna(row.get("scorer")) else str(row.get("scorer"))
    question = template["text"].format(
        home=home,
        away=away,
        date=_format_question_date(str(row["date_standardized"]), date_format),
        player=player,
    )

    answer = str(row["answer"])
    if intent == "match_score":
        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        if reverse:
            answer = f"{original_away} {away_score}-{home_score} {original_home}."
        else:
            answer = f"{original_home} {home_score}-{away_score} {original_away}."

    rendered = dict(row)
    rendered.update(
        {
            "semantic_question_id": semantic_id,
            "question_id": f"V2_{evaluation_track.upper()}_{semantic_id}",
            "question": question,
            "answer": answer,
            "template_id": template["template_id"],
            "template_family": template["template_family"],
            "template_seen_in_training": seen_in_training,
            "evaluation_track": evaluation_track,
            "question_variant": "seen_template" if seen_in_training else "unseen_template",
            "date_format": date_format,
            "team_order": "reversed" if reverse else "original",
        }
    )
    return rendered


def diversify_questions(
    dataframe: pd.DataFrame,
    *,
    evaluation_track: str = "standard",
    seen_in_training: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Render one deterministic linguistic variant for each semantic QA row."""
    registry = SEEN_TEMPLATES if seen_in_training else UNSEEN_TEMPLATES
    rendered = [
        _render_v2_row(
            row,
            templates=registry,
            evaluation_track=evaluation_track,
            seen_in_training=seen_in_training,
            random_state=random_state,
        )
        for row in dataframe.to_dict(orient="records")
    ]
    result = pd.DataFrame(rendered)
    leading = [
        "question_id", "semantic_question_id", "match_id", "intent", "question", "answer",
        "template_id", "template_family", "template_seen_in_training",
        "evaluation_track", "question_variant", "date_format", "team_order",
    ]
    return result[leading + [column for column in result.columns if column not in leading]]


def validate_v2_datasets(
    standard_splits: Mapping[str, pd.DataFrame], challenge_test: pd.DataFrame
) -> pd.DataFrame:
    """Validate semantic preservation, template separation, and match isolation."""
    for split_name, dataframe in standard_splits.items():
        validate_qa_dataset(dataframe)
        if not dataframe["template_seen_in_training"].eq(True).all():
            raise ValueError(f"{split_name} contains a template not marked as seen.")
        if not dataframe["evaluation_track"].eq("standard").all():
            raise ValueError(f"{split_name} contains an incorrect evaluation track.")

    validate_qa_dataset(challenge_test)
    if not challenge_test["template_seen_in_training"].eq(False).all():
        raise ValueError("Challenge rows must use unseen templates.")
    if not challenge_test["evaluation_track"].eq("challenge").all():
        raise ValueError("Challenge rows have an incorrect evaluation track.")

    train_templates = set(standard_splits["train"]["template_id"])
    challenge_templates = set(challenge_test["template_id"])
    if train_templates & challenge_templates:
        raise ValueError("Challenge templates overlap with training templates.")

    standard_test = standard_splits["test"]
    if set(standard_test["semantic_question_id"]) != set(
        challenge_test["semantic_question_id"]
    ):
        raise ValueError("Standard and challenge tests do not cover identical semantics.")
    if set(standard_test["match_id"]) != set(challenge_test["match_id"]):
        raise ValueError("Standard and challenge tests do not cover identical matches.")

    train_matches = set(standard_splits["train"]["match_id"])
    validation_matches = set(standard_splits["validation"]["match_id"])
    test_matches = set(standard_test["match_id"])
    if train_matches & validation_matches or train_matches & test_matches or validation_matches & test_matches:
        raise ValueError("Match leakage exists in Version 2.")

    return pd.DataFrame(
        [
            {
                "dataset": name,
                "rows": len(dataframe),
                "unique_matches": dataframe["match_id"].nunique(),
                "templates": dataframe["template_id"].nunique(),
                "seen_templates": bool(dataframe["template_seen_in_training"].all()),
            }
            for name, dataframe in {**standard_splits, "challenge_test": challenge_test}.items()
        ]
    )


def export_v2_data(
    standard_splits: Mapping[str, pd.DataFrame],
    challenge_test: pd.DataFrame,
    export_directory: str | Path,
) -> dict[str, Path]:
    """Export Version 2 standard splits, challenge test, and combined master."""
    export_directory = Path(export_directory)
    split_directory = export_directory / "splits"
    split_directory.mkdir(parents=True, exist_ok=True)
    standard_master = pd.concat(standard_splits.values(), ignore_index=True)
    paths: dict[str, Path] = {}

    datasets = {
        "master": (standard_master, export_directory / "football_qa_v2_master"),
        "train": (standard_splits["train"], split_directory / "football_qa_v2_train"),
        "validation": (standard_splits["validation"], split_directory / "football_qa_v2_validation"),
        "test": (standard_splits["test"], split_directory / "football_qa_v2_test"),
        "challenge_test": (challenge_test, split_directory / "football_qa_v2_challenge_test"),
    }
    for name, (dataframe, base_path) in datasets.items():
        for extension in ("csv", "jsonl"):
            path = base_path.with_suffix(f".{extension}")
            _write_dataframe(dataframe, path)
            paths[f"{name}_{extension}"] = path
            reloaded = (
                pd.read_csv(path, low_memory=False)
                if extension == "csv"
                else pd.read_json(path, lines=True)
            )
            if len(reloaded) != len(dataframe):
                raise ValueError(f"Exported Version 2 {name} row count is incorrect.")
            if not reloaded["question_id"].is_unique:
                raise ValueError(f"Exported Version 2 {name} question IDs are not unique.")
            if not reloaded[["question", "answer"]].notna().all().all():
                raise ValueError(f"Exported Version 2 {name} contains missing QA text.")
    return paths


def load_v2_splits(
    split_directory: str | Path, *, file_format: str = "csv"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Version 2 train, validation, standard-test, and challenge-test files."""
    if file_format not in {"csv", "jsonl"}:
        raise ValueError("file_format must be 'csv' or 'jsonl'.")
    split_directory = Path(split_directory)
    loaded = []
    for name in ("train", "validation", "test", "challenge_test"):
        path = split_directory / f"football_qa_v2_{name}.{file_format}"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared Version 2 split not found: {path}")
        loaded.append(
            pd.read_csv(path, low_memory=False)
            if file_format == "csv"
            else pd.read_json(path, lines=True)
        )
    return loaded[0], loaded[1], loaded[2], loaded[3]


def run_v2_pipeline(
    data_directory: str | Path,
    export_directory: str | Path,
    *,
    enforce_reference_counts: bool = True,
    random_state: int = RANDOM_SEED,
) -> dict[str, object]:
    """Build Version 1 semantics, then produce the diversified Version 2 datasets."""
    raw = load_raw_data(data_directory)
    semantic_master = build_qa_dataset(
        *raw, enforce_reference_counts=enforce_reference_counts
    )
    semantic_splits = create_match_level_splits(
        semantic_master, random_state=random_state
    )
    standard_splits = {
        name: diversify_questions(
            dataframe,
            evaluation_track="standard",
            seen_in_training=True,
            random_state=random_state,
        )
        for name, dataframe in semantic_splits.items()
    }
    challenge_test = diversify_questions(
        semantic_splits["test"],
        evaluation_track="challenge",
        seen_in_training=False,
        random_state=random_state,
    )
    validation_summary = validate_v2_datasets(standard_splits, challenge_test)
    paths = export_v2_data(standard_splits, challenge_test, export_directory)
    return {
        "semantic_master": semantic_master,
        "standard_splits": standard_splits,
        "challenge_test": challenge_test,
        "validation_summary": validation_summary,
        "paths": paths,
    }


def run_pipeline(
    data_directory: str | Path,
    export_directory: str | Path,
    *,
    enforce_reference_counts: bool = True,
    random_state: int = RANDOM_SEED,
) -> dict[str, object]:
    """Run the full shared pipeline and return its in-memory artifacts."""
    raw = load_raw_data(data_directory)
    master = build_qa_dataset(*raw, enforce_reference_counts=enforce_reference_counts)
    splits = create_match_level_splits(master, random_state=random_state)
    paths = export_prepared_data(master, splits, export_directory)
    return {"master": master, "splits": splits, "paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing the three raw CSV files.")
    parser.add_argument("--export-dir", required=True, help="Destination for master and split exports.")
    parser.add_argument(
        "--allow-different-counts",
        action="store_true",
        help="Allow newer raw dataset snapshots whose counts differ from the verified notebook.",
    )
    parser.add_argument(
        "--version",
        choices=("1", "2"),
        default="2",
        help="Dataset version to build (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.version == "1":
        artifacts = run_pipeline(
            args.data_dir,
            args.export_dir,
            enforce_reference_counts=not args.allow_different_counts,
        )
        master = artifacts["master"]
        splits = artifacts["splits"]
        assert isinstance(master, pd.DataFrame)
        assert isinstance(splits, dict)
        print(f"Version 1 master rows: {len(master):,}")
        for name, dataframe in splits.items():
            print(f"{name.title()} rows: {len(dataframe):,}")
    else:
        artifacts = run_v2_pipeline(
            args.data_dir,
            args.export_dir,
            enforce_reference_counts=not args.allow_different_counts,
        )
        standard_splits = artifacts["standard_splits"]
        challenge_test = artifacts["challenge_test"]
        assert isinstance(standard_splits, dict)
        assert isinstance(challenge_test, pd.DataFrame)
        print("Version 2 standard splits:")
        for name, dataframe in standard_splits.items():
            print(f"{name.title()} rows: {len(dataframe):,}")
        print(f"Challenge test rows: {len(challenge_test):,}")
    print("Shared football QA pipeline completed and validated.")


if __name__ == "__main__":
    main()
