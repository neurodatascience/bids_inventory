"""Tests of `bids_inventory neurobagel`, without network access."""

import json
import math

import pandas as pd
import pytest

# needs the 'update' extra
pytest.importorskip("requests")

from bids_inventory import _neurobagel  # noqa: E402
from bids_inventory._cli import cli  # noqa: E402
from bids_inventory._neurobagel import (  # noqa: E402
    NOT_ANNOTATED,
    check_annotations,
    parse_age,
    repos_to_process,
    standardize,
)


@pytest.mark.parametrize(
    "value, age_format, expected",
    [
        ("26", "nb:FromInt", 26),
        ("12.5", "nb:FromFloat", 12.5),
        ("12,5", "nb:FromEuro", 12.5),
        ("90+", "nb:FromBounded", 90),
        ("20-30", "nb:FromRange", 25),
        ("P20Y6M", "nb:FromISO8601", 20.5),
        ("42", "", 42),
    ],
)
def test_parse_age(value, age_format, expected):
    assert parse_age(value, age_format, set()) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["n/a", "", "unknown", "P", "20 years"])
def test_parse_age_missing_or_invalid(value):
    assert math.isnan(parse_age(value, "nb:FromFloat", {"unknown"}))


def annotation(term, **extra):
    return {"Annotations": {"IsAbout": {"TermURL": term, "Label": ""}, **extra}}


ANNOTATIONS = {
    "participant_id": annotation("nb:ParticipantID"),
    "age": annotation("nb:Age", Format={"TermURL": "nb:FromInt"}, MissingValues=["n/a"]),
    # the column is found from its annotation, not its name
    "gender": annotation(
        "nb:Sex",
        Levels={"F": {"TermURL": "snomed:248152002"}, "M": {"TermURL": "snomed:248153007"}},
        MissingValues=["n/a"],
    ),
}


def test_standardize():
    participants = pd.DataFrame(
        {
            "participant_id": ["sub-01", "sub-02", "sub-03", "sub-04"],
            "age": ["26", "n/a", "31", "40"],
            "gender": ["F", "M ", "n/a", "D"],
        }
    )
    rows, columns = standardize(participants, ANNOTATIONS)

    assert columns["sex"] == "gender"
    assert rows["participant_id"].tolist() == ["sub-01", "sub-02", "sub-03", "sub-04"]
    assert rows["age"].tolist()[0] == 26 and math.isnan(rows["age"].tolist()[1])
    # spaces around values are ignored, values not in the levels are unknown
    assert rows["sex"].tolist()[:2] == ["female", "male"]
    assert rows["sex"].isna().tolist() == [False, False, True, True]
    assert check_annotations(ANNOTATIONS, columns) == []


def test_standardize_without_participant_id():
    rows, _ = standardize(pd.DataFrame({"age": ["26"]}), {"age": ANNOTATIONS["age"]})
    assert rows is None


def test_check_annotations_sex_without_levels():
    """For example a participants.tsv with the age and sex columns swapped."""
    annotations = {
        "participant_id": annotation("nb:ParticipantID"),
        "sex": annotation("nb:Sex", Format={"TermURL": "nb:FromFloat"}),
    }
    participants = pd.DataFrame({"participant_id": ["sub-01"], "sex": ["15.1"]})
    _, columns = standardize(participants, annotations)
    assert check_annotations(annotations, columns) == [
        "sex column annotated without levels",
        "sex column annotated with a numerical format",
    ]


def test_repos_to_process():
    repos = [{"name": x} for x in ["ds000001", "ds000002", "ds000003", "ds000004"]]
    previous = pd.DataFrame(
        {
            "dataset_id": ["ds000001", "ds000002", "ds000003"],
            "reason": ["", NOT_ANNOTATED, "no participants.json"],
        }
    )
    # new repositories, and those that may have been annotated since
    names = [r["name"] for r in repos_to_process(repos, previous, reindex=False)]
    assert names == ["ds000002", "ds000004"]
    assert repos_to_process(repos, previous, reindex=True) == repos


@pytest.fixture
def fake_github(tmp_path, monkeypatch):
    """Fake repositories: ds000001 annotated, ds000002 not annotated yet."""
    repos = {
        "ds000001": {"annotated": True},
        "ds000002": {"annotated": False},
    }
    tsv = "participant_id\tage\tgender\nsub-01\t26\tF\nsub-02\t24\tM\n"

    def list_repos(session):
        return [
            {"name": name, "default_branch": "main", "pushed_at": "2026-01-01T00:00:00Z"}
            for name in repos
        ]

    def download(session, url, cache_dir, repo):
        return json.dumps(ANNOTATIONS) if url.endswith(".json") else tsv

    monkeypatch.setattr(_neurobagel, "list_repos", list_repos)
    monkeypatch.setattr(_neurobagel, "download", download)
    monkeypatch.setattr(
        _neurobagel,
        "annotated_by_neurobagel",
        lambda session, repo, cache_dir: repos[repo["name"]]["annotated"],
    )
    monkeypatch.setattr(_neurobagel, "DATASETS_FILE", tmp_path / "neurobagel_datasets.tsv")
    monkeypatch.setattr(_neurobagel, "PARTICIPANTS_FILE", tmp_path / "neurobagel_participants.tsv")
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    return repos, tmp_path


def test_cli_neurobagel(fake_github, monkeypatch):
    repos, tmp_path = fake_github
    processed = []
    process_repo = _neurobagel.process_repo
    monkeypatch.setattr(
        _neurobagel,
        "process_repo",
        lambda session, repo, cache_dir: (
            processed.append(repo["name"]) or process_repo(session, repo, cache_dir)
        ),
    )

    cli(["bids_inventory", "neurobagel", "--cache-dir", str(tmp_path / "cache")])

    datasets = pd.read_csv(tmp_path / "neurobagel_datasets.tsv", sep="\t", keep_default_na=False)
    participants = pd.read_csv(tmp_path / "neurobagel_participants.tsv", sep="\t")
    assert datasets["status"].tolist() == ["used", "skipped"]
    assert datasets["reason"].tolist() == ["", NOT_ANNOTATED]
    assert participants["sex"].tolist() == ["female", "male"]
    assert processed == ["ds000001", "ds000002"]

    # ds000002 gets annotated: only it is processed again
    repos["ds000002"]["annotated"] = True
    processed.clear()
    cli(["bids_inventory", "neurobagel", "--cache-dir", str(tmp_path / "cache")])
    assert processed == ["ds000002"]
    participants = pd.read_csv(tmp_path / "neurobagel_participants.tsv", sep="\t")
    assert participants["dataset_id"].tolist() == ["ds000001"] * 2 + ["ds000002"] * 2

    # --reindex processes everything again
    processed.clear()
    cli(["bids_inventory", "neurobagel", "--reindex", "--cache-dir", str(tmp_path / "cache")])
    assert processed == ["ds000001", "ds000002"]


def test_cli_neurobagel_token_file_not_found(tmp_path):
    with pytest.raises(SystemExit):
        cli(["bids_inventory", "neurobagel", "--token-file", str(tmp_path / "missing.txt")])
