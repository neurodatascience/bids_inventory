"""Combine the participants.tsv of the datasets annotated by Neurobagel.

The repositories of https://github.com/OpenNeuroDatasets-JSONLD have a participants.json
annotated by Neurobagel (commits "[bot] Update participants.json"):
each annotated column says what it is about (``IsAbout``: ``nb:Age``, ``nb:Sex``...),
how its values are formatted, and which values are missing.

This script uses these annotations to combine the participants.tsv of all these datasets
in a single TSV with the age (in years) and the sex (female, male, other)
standardized across datasets.

Usage:

    python tools/combine_neurobagel_participants.py --token-file ~/tokens/github.txt

Outputs (in --output-dir, default: the current directory):

- neurobagel_participants.tsv: one row per participant (and session) of each dataset
  (dataset_id, participant_id, session_id, age, sex, and their raw values);
- neurobagel_datasets.tsv: one row per repository, with why it was used or not.

Downloaded files are cached in --cache-dir (default: neurobagel_cache):
run the script again to only download what changed.
"""

import argparse
import io
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ORGANIZATION = "OpenNeuroDatasets-JSONLD"
API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"

# message of the commits that add the Neurobagel annotations
NEUROBAGEL_COMMIT = "Update participants.json"
NEUROBAGEL_BOT = "neurobagel-bot[bot]"

SEX = {
    "snomed:248152002": "female",
    "snomed:248153007": "male",
    "snomed:32570681000036106": "other",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Text file with a GitHub token (the GitHub API is rate limited without one).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, default=Path("neurobagel_cache"))
    parser.add_argument(
        "--datasets", nargs="+", help="Only these datasets (for example: ds000001 ds000002)."
    )
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel downloads.")
    args = parser.parse_args(argv)

    session = requests.Session()
    if args.token_file:
        token = args.token_file.expanduser().read_text().strip()
        session.headers["Authorization"] = f"Bearer {token}"

    repos = list_repos(session)
    if args.datasets:
        repos = [r for r in repos if r["name"] in set(args.datasets)]
    print(f"{len(repos)} repositories in {ORGANIZATION}")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(
            executor.map(lambda repo: process_repo(session, repo, args.cache_dir), repos)
        )

    datasets = pd.DataFrame([summary for summary, _ in results])
    for column in ["nb_participants", "nb_age", "nb_sex"]:
        if column in datasets:
            datasets[column] = datasets[column].astype("Int64")
    participants = pd.concat(
        [rows for _, rows in results if rows is not None and len(rows)], ignore_index=True
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets.to_csv(args.output_dir / "neurobagel_datasets.tsv", sep="\t", index=False)
    participants.to_csv(args.output_dir / "neurobagel_participants.tsv", sep="\t", index=False)

    used = datasets[datasets["status"] == "used"]
    print(datasets["status"].value_counts().to_string())
    print(
        f"{len(participants)} participants from {len(used)} datasets "
        f"(age known for {participants['age'].notna().sum()}, "
        f"sex known for {participants['sex'].notna().sum()})"
    )
    print(f"Saved to {args.output_dir / 'neurobagel_participants.tsv'}")


def list_repos(session: requests.Session) -> list[dict[str, Any]]:
    repos, page = [], 1
    while True:
        response = session.get(
            f"{API}/orgs/{ORGANIZATION}/repos", params={"per_page": 100, "page": page}
        )
        response.raise_for_status()
        if not (batch := response.json()):
            break
        repos.extend(batch)
        page += 1
    return sorted(
        (r for r in repos if r["name"].startswith("ds") and r["size"] > 0),
        key=lambda r: r["name"],
    )


def process_repo(
    session: requests.Session, repo: dict[str, Any], cache_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Return a summary of the repository, and its standardized participants if it is used."""
    name, branch = repo["name"], repo["default_branch"]
    summary: dict[str, Any] = {"dataset_id": name, "status": "", "reason": ""}
    try:
        sidecar = download(
            session,
            f"{RAW}/{ORGANIZATION}/{name}/{branch}/participants.json",
            cache_dir / name,
            repo,
        )
        tsv = download(
            session,
            f"{RAW}/{ORGANIZATION}/{name}/{branch}/participants.tsv",
            cache_dir / name,
            repo,
        )
        if sidecar is None or tsv is None:
            missing = [
                f
                for f, c in [("participants.json", sidecar), ("participants.tsv", tsv)]
                if c is None
            ]
            return {
                **summary,
                "status": "skipped",
                "reason": f"no {' and no '.join(missing)}",
            }, None

        if not annotated_by_neurobagel(session, repo, cache_dir / name):
            return {
                **summary,
                "status": "skipped",
                "reason": "participants.json not updated by Neurobagel",
            }, None

        annotations = json.loads(sidecar)
        participants = pd.read_csv(io.StringIO(tsv), sep="\t", dtype=str, keep_default_na=False)
        rows, columns = standardize(participants, annotations)
        summary.update({f"{k}_column": v for k, v in columns.items()})
        summary["notes"] = "; ".join(check_annotations(annotations, columns))
        if rows is None:
            return {
                **summary,
                "status": "skipped",
                "reason": "no annotated participant ID column",
            }, None
        rows.insert(0, "dataset_id", name)
        if (rows["age"] < 0).any():
            # for example ds001365, where all the columns are standardized (z-scores)
            notes = [summary["notes"]] if summary["notes"] else []
            summary["notes"] = "; ".join([*notes, "negative ages: probably not in years"])
        summary.update(
            status="used",
            nb_participants=rows["participant_id"].nunique(),
            nb_age=int(rows["age"].notna().sum()),
            nb_sex=int(rows["sex"].notna().sum()),
        )
        return summary, rows
    except Exception as e:
        return {**summary, "status": "error", "reason": repr(e)[:200]}, None


def download(
    session: requests.Session, url: str, cache_dir: Path, repo: dict[str, Any]
) -> str | None:
    """Download a file of a repository, or return None if it does not exist.

    Cached until the repository changes (its last push date).
    """
    cache = cache_dir / f"{Path(url).name}.{repo['pushed_at'].replace(':', '')}"
    missing = cache.with_name(cache.name + ".missing")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    if missing.exists():
        return None
    response = session.get(url, timeout=60)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{Path(url).name}.*"):
        old.unlink()
    if response.status_code == 404:
        missing.touch()
        return None
    response.raise_for_status()
    response.encoding = "utf-8"
    cache.write_text(response.text, encoding="utf-8")
    return response.text


def annotated_by_neurobagel(
    session: requests.Session, repo: dict[str, Any], cache_dir: Path
) -> bool:
    """Return True if participants.json was updated by a Neurobagel commit."""
    cache = cache_dir / f"commits.{repo['pushed_at'].replace(':', '')}.json"
    if cache.exists():
        commits = json.loads(cache.read_text())
    else:
        response = session.get(
            f"{API}/repos/{ORGANIZATION}/{repo['name']}/commits",
            params={"path": "participants.json", "per_page": 100},
            timeout=60,
        )
        response.raise_for_status()
        commits = [
            {"author": c["commit"]["author"]["name"], "message": c["commit"]["message"]}
            for c in response.json()
        ]
        for old in cache_dir.glob("commits.*.json"):
            old.unlink()
        cache.write_text(json.dumps(commits))
    return any(NEUROBAGEL_COMMIT in c["message"] or c["author"] == NEUROBAGEL_BOT for c in commits)


def find_column(annotations: dict[str, Any], term: str) -> str | None:
    """Return the column annotated as being about ``term`` (for example nb:Age)."""
    for column, description in annotations.items():
        about = (description or {}).get("Annotations", {}).get("IsAbout", {})
        if isinstance(about, dict) and about.get("TermURL") == term:
            return column
    return None


def standardize(
    participants: pd.DataFrame, annotations: dict[str, Any]
) -> tuple[pd.DataFrame | None, dict[str, str | None]]:
    """Return the participants with their standardized age and sex, and the columns used."""
    columns = {
        "participant_id": find_column(annotations, "nb:ParticipantID"),
        "session_id": find_column(annotations, "nb:Session"),
        "age": find_column(annotations, "nb:Age"),
        "sex": find_column(annotations, "nb:Sex"),
    }
    columns = {k: v if v in participants.columns else None for k, v in columns.items()}
    if columns["participant_id"] is None:
        return None, columns

    rows = pd.DataFrame({"participant_id": participants[columns["participant_id"]]})
    if columns["session_id"]:
        rows["session_id"] = participants[columns["session_id"]]

    age_column, sex_column = columns["age"], columns["sex"]
    if age_column:
        age_annotations = annotations[age_column]["Annotations"]
        age_format = (age_annotations.get("Format") or {}).get("TermURL", "")
        missing = set(age_annotations.get("MissingValues") or [])
        rows["age"] = [parse_age(x, age_format, missing) for x in participants[age_column]]
        rows["age_raw"] = participants[age_column]
        rows["age_format"] = age_format
    else:
        rows["age"] = math.nan
    if sex_column:
        sex_annotations = annotations[sex_column]["Annotations"]
        levels = sex_annotations.get("Levels") or {}
        missing = set(sex_annotations.get("MissingValues") or [])
        rows["sex"] = [
            None
            if x in missing or x.strip() in missing
            else SEX.get((levels.get(x) or levels.get(x.strip()) or {}).get("TermURL", ""))
            for x in participants[sex_column]
        ]
        rows["sex_raw"] = participants[sex_column]
    else:
        rows["sex"] = None
    return rows, columns


def check_annotations(annotations: dict[str, Any], columns: dict[str, str | None]) -> list[str]:
    """Return the problems found in the annotations of the age and sex columns."""
    notes = []
    if columns["sex"]:
        sex = annotations[columns["sex"]]["Annotations"]
        if not sex.get("Levels"):
            # for example a sex column that contains ages (columns swapped in participants.tsv)
            notes.append("sex column annotated without levels")
        if sex.get("Format"):
            notes.append("sex column annotated with a numerical format")
    if columns["age"] and annotations[columns["age"]]["Annotations"].get("Levels"):
        notes.append("age column annotated with levels")
    return notes


def parse_age(value: str, age_format: str, missing: set[str]) -> float:
    """Convert an age to years, following its Neurobagel format.

    Returns NaN for missing values and values that cannot be converted.
    """
    value = value.strip()
    if value in missing or value.lower() in {"", "n/a", "na", "nan"}:
        return math.nan
    try:
        if age_format == "nb:FromEuro":
            return float(value.replace(",", "."))
        if age_format == "nb:FromBounded":
            # for example "90+": the bound
            return float(value.rstrip("+").strip())
        if age_format == "nb:FromRange":
            # for example "20-30": the middle of the range
            low, high = re.split(r"\s*-\s*", value, maxsplit=1)
            return (float(low) + float(high)) / 2
        if age_format == "nb:FromISO8601":
            # for example "P20Y6M"
            match = re.fullmatch(
                r"P(?:(\d+(?:\.\d+)?)Y)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)W)?(?:(\d+(?:\.\d+)?)D)?",
                value,
            )
            if not match or not any(match.groups()):
                return math.nan
            years, months, weeks, days = (float(x) if x else 0.0 for x in match.groups())
            return years + months / 12 + weeks / 52.1775 + days / 365.25
        # nb:FromFloat, nb:FromInt, or no format
        return float(value)
    except ValueError:
        return math.nan


if __name__ == "__main__":
    sys.exit(main())
