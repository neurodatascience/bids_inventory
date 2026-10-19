"""Tests of `bids_inventory neurobagel`, without network access."""

import json
import math

import pandas as pd
import pytest

# needs the 'update' extra
pytest.importorskip("requests")
pytest.importorskip("datalad")

from bids_inventory import _neurobagel  # noqa: E402
from bids_inventory._cli import cli  # noqa: E402
from bids_inventory._neurobagel import (  # noqa: E402
    NOT_ANNOTATED,
    add_species,
    check_annotations,
    find_species,
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


@pytest.mark.parametrize(
    "column, values, expected",
    [
        ("species", ["homo sapien", "Homo sapiens ", "n/a"], "homo sapiens"),
        ("Species", ["rat", "mus musculus"], "mus musculus, rattus norvegicus"),
        ("species", ["", ""], ""),
        ("age", ["26", "30"], ""),
    ],
)
def test_find_species(column, values, expected):
    assert find_species(pd.DataFrame({column: values})) == expected


def test_add_species(tmp_path, monkeypatch):
    species_file = tmp_path / "species.tsv"
    species_file.write_text("dataset_id\tspecies\tcomment\nds000002\tovis aries\tsheep\n")
    monkeypatch.setattr(_neurobagel, "SPECIES_FILE", species_file)
    datasets = pd.DataFrame(
        {"dataset_id": ["ds000001", "ds000002", "ds000003"], "species": ["mus musculus", "", None]}
    )
    assert add_species(datasets)["species"].tolist() == ["mus musculus", "ovis aries", ""]
    # the outputs of a run before the species were added
    assert add_species(datasets[["dataset_id"]])["species"].tolist() == ["", "ovis aries", ""]


def test_repos_to_process():
    repos = [{"name": x} for x in ["ds000001", "ds000002", "ds000003", "ds000004", "ds000005"]]
    previous = pd.DataFrame(
        {
            "dataset_id": ["ds000001", "ds000002", "ds000003", "ds000005"],
            "status": ["used", "skipped", "skipped", "error"],
            "reason": ["", NOT_ANNOTATED, "no participants.json", "ConnectionError()"],
        }
    )
    # new repositories, those that may have been annotated since, and those that failed
    names = [r["name"] for r in repos_to_process(repos, previous, reindex=False)]
    assert names == ["ds000002", "ds000004", "ds000005"]
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
            if name != "node"
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
    monkeypatch.setattr(_neurobagel, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_neurobagel, "DATASETS_FILE", tmp_path / "neurobagel_datasets.tsv")
    monkeypatch.setattr(_neurobagel, "PARTICIPANTS_FILE", tmp_path / "neurobagel_participants.tsv")
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    # a Neurobagel node with one dataset
    node_rows = _neurobagel.node_participants(
        "CORR_XHCUMS",
        [
            {
                "sub_id": "sub-01",
                "session_type": "http://neurobagel.org/vocab/PhenotypicSession",
                "age": 30.0,
                "sex": "http://purl.bioontology.org/ontology/SNOMEDCT/248152002",
            }
        ],
    )
    node = {"available": True}

    def fake_fetch_node(session, node_name, node_url):
        if not node["available"]:
            raise _neurobagel.requests.ConnectionError("node down")
        summary = {
            "dataset_id": "CORR_XHCUMS",
            "status": "used",
            "reason": "",
            "source": f"Neurobagel node: {node_name}",
            "nb_participants": 1,
            "nb_age": 1,
            "nb_sex": 1,
        }
        return [summary], node_rows

    monkeypatch.setattr(_neurobagel, "fetch_node", fake_fetch_node)
    monkeypatch.setattr(_neurobagel, "NODES", {"INDI": "https://node"})
    repos["node"] = node
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

    cli(["bids_inventory", "neurobagel"])

    datasets = pd.read_csv(tmp_path / "neurobagel_datasets.tsv", sep="\t", keep_default_na=False)
    participants = pd.read_csv(tmp_path / "neurobagel_participants.tsv", sep="\t")
    # the datasets of the Neurobagel nodes are added
    assert datasets["dataset_id"].tolist() == ["CORR_XHCUMS", "ds000001", "ds000002"]
    assert datasets["source"].tolist() == [
        "Neurobagel node: INDI",
        "OpenNeuroDatasets-JSONLD",
        "OpenNeuroDatasets-JSONLD",
    ]
    assert datasets["status"].tolist() == ["used", "used", "skipped"]
    assert datasets["reason"].tolist() == ["", "", NOT_ANNOTATED]
    # ds000001 has 16 sub-* folders, but 2 participants in this participants.tsv
    assert datasets["participants_mismatch"].tolist()[1:] == [True, False]
    assert participants["sex"].tolist() == ["female", "female", "male"]
    assert processed == ["ds000001", "ds000002"]

    # ds000002 gets annotated: only it is processed again
    repos["ds000002"]["annotated"] = True
    processed.clear()
    cli(["bids_inventory", "neurobagel"])
    assert processed == ["ds000002"]
    participants = pd.read_csv(tmp_path / "neurobagel_participants.tsv", sep="\t")
    assert (
        participants["dataset_id"].tolist() == ["CORR_XHCUMS"] + ["ds000001"] * 2 + ["ds000002"] * 2
    )

    # a node that cannot be reached keeps the rows of the previous run
    repos["node"]["available"] = False
    cli(["bids_inventory", "neurobagel"])
    datasets = pd.read_csv(tmp_path / "neurobagel_datasets.tsv", sep="\t", keep_default_na=False)
    participants = pd.read_csv(tmp_path / "neurobagel_participants.tsv", sep="\t")
    assert datasets["source"].tolist()[0] == "Neurobagel node: INDI"
    assert participants["dataset_id"].tolist().count("CORR_XHCUMS") == 1

    # --reindex processes everything again
    processed.clear()
    cli(["bids_inventory", "neurobagel", "--reindex"])
    assert processed == ["ds000001", "ds000002"]


def test_cli_neurobagel_config_not_found(tmp_path):
    with pytest.raises(SystemExit):
        cli(["bids_inventory", "neurobagel", "--config", str(tmp_path / "missing.yml")])


def test_cli_neurobagel_token_from_config(fake_github, tmp_path, monkeypatch):
    """The token is read from auth.token_file, as for 'bids_inventory update'."""
    from bids_inventory._update import CONFIG_ENV_VAR, config

    monkeypatch.delenv("GITHUB_TOKEN")
    token_file = tmp_path / "token.txt"
    token_file.write_text("token_from_file\n")
    user_config = tmp_path / "config.yml"
    user_config.write_text(f"auth:\n    token_file: {token_file}\n")
    tokens = []
    monkeypatch.setattr(
        _neurobagel,
        "update_neurobagel",
        lambda token, **kwargs: tokens.append(token),
    )
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    config.cache_clear()

    cli(["bids_inventory", "neurobagel", "--config", str(user_config)])

    config.cache_clear()
    assert tokens == ["token_from_file"]


def test_add_subject_counts(monkeypatch):
    """Datasets used with a different number of participants and of sub-* folders are flagged."""
    listing = pd.DataFrame({"name": ["ds1", "ds2", "ds3"], "nb_subjects": [10, 20, 30]})
    monkeypatch.setattr(_neurobagel, "known_datasets_df", lambda: listing)
    datasets = pd.DataFrame(
        {
            "dataset_id": ["ds1", "ds2", "ds3", "ds4"],
            "status": ["used", "used", "skipped", "used"],
            "nb_participants": ["10", "25", "", "5"],
        }
    )

    datasets = _neurobagel.add_subject_counts(datasets)

    assert datasets["nb_subjects"].tolist()[:3] == [10, 20, 30]
    # ds4 is not in the inventory
    assert pd.isna(datasets["nb_subjects"].tolist()[3])
    assert datasets["participants_mismatch"].tolist() == [False, True, False, False]


def test_node_participants():
    """One row per participant, from its phenotypic session."""
    sessions = [
        {
            "sub_id": "sub-01",
            "session_type": "http://neurobagel.org/vocab/PhenotypicSession",
            "age": 14.0,
            "sex": "http://purl.bioontology.org/ontology/SNOMEDCT/248153007",
        },
        {"sub_id": "sub-01", "session_type": "http://neurobagel.org/vocab/ImagingSession"},
        {
            "sub_id": "sub-02",
            "session_type": "http://neurobagel.org/vocab/PhenotypicSession",
            "age": None,
            "sex": None,
        },
    ]
    rows = _neurobagel.node_participants("ABIDE_NYU", sessions)
    assert rows["participant_id"].tolist() == ["sub-01", "sub-02"]
    assert rows["age"].tolist()[0] == 14.0 and math.isnan(rows["age"].tolist()[1])
    assert rows["sex"].tolist()[0] == "male" and pd.isna(rows["sex"].tolist()[1])
    assert rows["sex_raw"].tolist() == ["snomed:248153007", ""]


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://datasets.datalad.org/corr/RawDataBIDS/XHCUMS", "CORR_XHCUMS"),
        ("https://datasets.datalad.org/abide2/RawData/KKI_1/", "ABIDE2_KKI_1"),
        ("https://github.com/nemarDatasets/nm000103.git", "NEMAR_nm000103"),
        ("https://example.org/not/tracked", None),
        (None, None),
    ],
)
def test_inventory_name(url, expected):
    assert _neurobagel.inventory_name(url) == expected


def test_fetch_node(monkeypatch):
    """The datasets of a node, then the participants of these datasets."""

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return self.content

    datasets = [
        {
            "dataset_uuid": "a",
            "dataset_name": "corr-XHCUMS",
            "repository_url": "https://datasets.datalad.org/corr/RawDataBIDS/XHCUMS",
        },
        {
            "dataset_uuid": "b",
            "dataset_name": "Other",
            "repository_url": "https://example.org/other",
        },
        {
            "dataset_uuid": "c",
            "dataset_name": "abide-NYU",
            "repository_url": "https://datasets.datalad.org/abide/RawDataBIDS/NYU",
        },
    ]
    phenotypic = "http://neurobagel.org/vocab/PhenotypicSession"
    subjects = [
        {
            "dataset_uuid": "a",
            "subject_data": [
                {"sub_id": "sub-1", "session_type": phenotypic, "age": 20.0, "sex": None}
            ],
        },
        {"dataset_uuid": "b", "subject_data": []},
        {"dataset_uuid": "c", "subject_data": "protected"},
    ]
    posts = []

    class Session:
        def post(self, url, json, timeout):
            posts.append((url.rsplit("/", 1)[-1], json))
            return Response({"responses": datasets if url.endswith("/datasets") else subjects})

    summaries, participants = _neurobagel.fetch_node(Session(), "INDI", "https://node/")

    assert posts[1] == (
        "subjects",
        {"nodes": [{"node_url": "https://node/", "dataset_uuids": ["a", "b", "c"]}]},
    )
    assert [(x["dataset_id"], x["status"], x["reason"]) for x in summaries] == [
        ("CORR_XHCUMS", "used", ""),
        ("Other", "skipped", "not in the inventory"),
        ("ABIDE_NYU", "skipped", "participants not available"),
    ]
    assert participants["participant_id"].tolist() == ["sub-1"]
