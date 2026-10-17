import pandas as pd
import pytest

from bids_inventory import filter_data, known_datasets_df, load_inventory, wrangle_data


def test_wrangle_data():
    wrangle_data(known_datasets_df())


def test_filter_data():
    """The default configuration keeps all the datasets with at least one BIDS datatype.

    Datasets that are not valid BIDS can have none,
    for example with data files directly in the session folders.
    """
    df = wrangle_data(known_datasets_df())
    fitlered_df = filter_data(df, config=None)

    with_datatypes = df[df["datatypes"].apply(len) > 0]
    assert len(fitlered_df) == len(with_datatypes)


def test_filter_data_no_datasets():
    df = wrangle_data(known_datasets_df())
    fitlered_df = filter_data(df, config={"datatypes": ["foo"]})

    assert len(fitlered_df) == 0


def test_wrangle_data_has_derivatives():
    """Only datasets with a derivative URL should be flagged as having it."""
    df = known_datasets_df()
    wrangled = wrangle_data(df)
    for der in ["fmriprep", "freesurfer", "mriqc"]:
        assert wrangled[f"has_{der}"].sum() == df[der].notna().sum()
        assert wrangled[f"has_{der}"].sum() < len(wrangled)


def test_load_inventory():
    df = load_inventory()
    assert len(df) == len(known_datasets_df())
    assert {"has_fmriprep", "nb_tasks", "total_duration", "source"} <= set(df.columns)


def test_filter_data_species():
    df = wrangle_data(known_datasets_df())
    species = dict(zip(df["name"], df["species"]))
    assert species["ds004161"] == ["ovis aries"]
    assert species["ds000001"] == []

    humans = filter_data(df, config={"species": ["homo sapiens", "not specified"]})
    assert "ds004161" not in set(humans["name"])
    assert "ds000001" in set(humans["name"])
    # a dataset is kept only if all its species are selected
    assert "ds005605" not in set(filter_data(df, config={"species": "mus musculus"})["name"])
    assert "ds005605" in set(
        filter_data(df, config={"species": ["mus musculus", "rattus norvegicus"]})["name"]
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"task": "REST"}, ["ds1", "ds2"]),
        ({"task": "^rest$", "task_regex": True}, ["ds1"]),
        ({"author": "poldrack"}, ["ds1"]),
        ({"author": r"^gau, r", "author_regex": True}, ["ds2"]),
        ({"institution": "mcgill"}, ["ds2"]),
        ({"country": "^(france|canada)$", "country_regex": True}, ["ds1", "ds2"]),
        ({"country": "canada", "author": "poldrack"}, []),
        # "true" is a text to search, not a boolean
        ({"author": "true"}, []),
    ],
)
def test_filter_data_text(config, expected):
    df = pd.DataFrame(
        {
            "name": ["ds1", "ds2", "ds3"],
            "datatypes": [["func"]] * 3,
            "tasks": [["rest"], ["restingstate"], ["nback"]],
            "authors": [["Russell A. Poldrack"], ["Gau, R."], []],
            "institution_names": [["Université Paris-Saclay"], ["McGill University"], []],
            "countries": [["France"], ["Canada"], []],
        }
    )
    assert filter_data(df, config=config)["name"].tolist() == expected


def test_filter_data_invalid_regex():
    df = wrangle_data(known_datasets_df())
    with pytest.raises(ValueError, match="Invalid regular expression"):
        filter_data(df, config={"author": "(", "author_regex": True})


def test_wrangle_data_institutions():
    """The institutions are those of institutions.json, without the unlocated ones."""
    from bids_inventory.institutions import UNLOCATED, load_institutions

    df = wrangle_data(known_datasets_df())
    institutions = load_institutions()
    names = {x for names in df["institution_names"] for x in names}
    assert names and names <= set(institutions) - {UNLOCATED}
    countries = {x for countries in df["countries"] for x in countries}
    assert "Canada" in countries
