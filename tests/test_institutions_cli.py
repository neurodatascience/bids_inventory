"""Tests of `bids_inventory institutions`, without network access."""

import json

import pandas as pd
import pytest

# needs the 'institutions' extra
pytest.importorskip("geopy")
pytest.importorskip("rapidfuzz")

from bids_inventory import _cli, _institutions_update, institutions  # noqa: E402
from bids_inventory._cli import cli  # noqa: E402
from bids_inventory.institutions import UNLOCATED, new_institution  # noqa: E402

GENEVA = {
    "address": "Campus Biotech, Chemin des Mines 9, Genève, Suisse",
    "city": "Genève",
    "country": "Suisse",
    "latitude": 46.2,
    "longitude": 6.1,
}


@pytest.fixture(autouse=True)
def no_postcode_location(monkeypatch):
    """No network access: the ZIP codes are not found."""
    monkeypatch.setattr(
        _institutions_update, "_postcode_location", lambda geocode, postcode, country: None
    )


@pytest.fixture
def institutions_file(tmp_path, monkeypatch):
    """Return an institutions.json with one located institution, with a small inventory."""
    institutions_json = tmp_path / "institutions.json"
    geneva = new_institution(GENEVA)
    institutions_json.write_text(json.dumps({"Campus Biotech, Geneva": geneva}))
    monkeypatch.setattr(institutions, "INSTITUTIONS_FILE", institutions_json)

    inventory = pd.DataFrame(
        {
            "institutions": [
                ["Campus Biotech, Geneva"],
                ["Fondation Campus Biotech, Geneve, CH"],
                ["Nowhere Institute, Atlantis"],
                ["Somewhere else"],
            ],
            "nb_subjects": [10, 20, 30, 40],
        }
    )
    monkeypatch.setattr("bids_inventory.load_inventory", lambda: inventory)
    return institutions_json


class FakeLocation:
    def __init__(self, location):
        self.address = location["address"]
        self.latitude, self.longitude = location["latitude"], location["longitude"]
        self.raw = {
            "address": {"city": location["city"], "country": location["country"]},
            "place_rank": location.get("place_rank", 30),
        }


def test_update(institutions_file, monkeypatch):
    locations = {
        # same address as an institution already in the file
        "Fondation Campus Biotech, Geneve, CH": FakeLocation(GENEVA),
        "Somewhere else": FakeLocation({**GENEVA, "address": "Elsewhere", "city": "Bern"}),
    }
    calls = []

    def fake_get_location(geocode, address, country=None):
        calls.append(address)
        return locations.get(address)

    monkeypatch.setattr(_institutions_update, "_get_location", fake_get_location)

    cli(["bids_inventory", "institutions", "update"])

    saved = json.loads(institutions_file.read_text())
    assert saved["Campus Biotech, Geneva"]["AlternativeNames"] == [
        "Fondation Campus Biotech, Geneve, CH"
    ]
    assert saved["Somewhere else"]["City"] == "Bern"
    assert saved[UNLOCATED]["AlternativeNames"] == ["Nowhere Institute, Atlantis"]

    # running it again does not geocode anything
    calls.clear()
    cli(["bids_inventory", "institutions", "update"])
    assert calls == []


def test_update_can_resume(institutions_file, monkeypatch):
    calls = []

    def stop_after_one(geocode, address, country=None):
        if calls:
            raise KeyboardInterrupt
        calls.append(address)
        return None

    monkeypatch.setattr(_institutions_update, "_get_location", stop_after_one)
    with pytest.raises(KeyboardInterrupt):
        cli(["bids_inventory", "institutions", "update"])

    # the first address was saved before the interruption
    saved = json.loads(institutions_file.read_text())
    assert len(saved[UNLOCATED]["AlternativeNames"]) == 1


def test_suggest_and_apply_matches(institutions_file, tmp_path, monkeypatch):
    institutions_json = json.loads(institutions_file.read_text())
    unlocated = new_institution()
    unlocated["AlternativeNames"] = ["Campus Biotech Geneva"]
    institutions_json[UNLOCATED] = unlocated
    institutions_file.write_text(json.dumps(institutions_json))
    matches_file = tmp_path / "matches.tsv"

    cli(["bids_inventory", "institutions", "suggest", "--output", str(matches_file)])

    matches = pd.read_csv(matches_file, sep="\t", dtype=str, keep_default_na=False)
    pair = {"Campus Biotech, Geneva", "Campus Biotech Geneva"}
    assert any({r.institution_1, r.institution_2} == pair for r in matches.itertuples())

    matches["accept"] = "y"
    matches.to_csv(matches_file, sep="\t", index=False)
    cli(["bids_inventory", "institutions", "apply-matches", "--matches", str(matches_file)])

    saved = json.loads(institutions_file.read_text())
    assert "Campus Biotech Geneva" in saved["Campus Biotech, Geneva"]["AlternativeNames"]
    assert UNLOCATED not in saved


def test_apply_matches_missing_file(institutions_file, tmp_path):
    with pytest.raises(SystemExit):
        cli(["bids_inventory", "institutions", "apply-matches", "--matches", str(tmp_path / "x")])
    assert _cli  # the CLI module is importable without the extra


def test_update_in_the_country_of_the_address(institutions_file, monkeypatch):
    """Addresses with a country code are only searched in that country."""
    inventory = pd.DataFrame(
        {
            "institutions": [["IBMI, Brenneckestr._6_Magdeburg_District_DE_39118"]],
            "nb_subjects": [1],
        }
    )
    monkeypatch.setattr("bids_inventory.load_inventory", lambda: inventory)
    calls = []
    monkeypatch.setattr(
        _institutions_update,
        "_get_location",
        lambda geocode, address, country=None: calls.append((address, country)),
    )

    cli(["bids_inventory", "institutions", "update"])

    # "District" is left from the DICOM address template
    assert calls == [("IBMI, Brenneckestr. 6 Magdeburg DE 39118", "de")]


MAGDEBURG = {
    "address": "Brenneckestraße 6, Magdeburg, Deutschland",
    "city": "Magdeburg",
    "country": "Deutschland",
    "latitude": 52.1,
    "longitude": 11.6,
}


@pytest.fixture
def mislocated(tmp_path, monkeypatch):
    """institutions.json with an institution located in the wrong country."""
    institutions_json = tmp_path / "institutions.json"
    institutions_json.write_text(
        json.dumps(
            {
                "IBMI, Brenneckestr._6_Magdeburg_District_DE_39118": new_institution(
                    {**GENEVA, "address": "IBMI, Cuiabá, Brasil", "country": "Brasil"}
                ),
                "Campus Biotech, Chemin des Mines 9,Geneve,District,CH,1211": new_institution(
                    GENEVA
                ),
                "University of Geneva": new_institution(
                    {**GENEVA, "address": "Geneva House, Kingston, Canada", "country": "Canada"}
                ),
            }
        )
    )
    monkeypatch.setattr(institutions, "INSTITUTIONS_FILE", institutions_json)
    return institutions_json


def test_relocate_all(mislocated, monkeypatch):
    """Only the institutions found in the country of their address, but located elsewhere."""
    calls = []

    def fake_get_location(geocode, address, country=None):
        calls.append(country)
        return {"de": FakeLocation(MAGDEBURG), "ch": FakeLocation(GENEVA)}[country]

    monkeypatch.setattr(_institutions_update, "_get_location", fake_get_location)

    cli(["bids_inventory", "institutions", "relocate"])

    saved = json.loads(mislocated.read_text())
    assert saved["IBMI, Brenneckestr._6_Magdeburg_District_DE_39118"]["Country"] == "Deutschland"
    # no country code: not checked
    assert saved["University of Geneva"]["Country"] == "Canada"
    assert sorted(calls) == ["ch", "de"]


def test_relocate_with_country(mislocated, monkeypatch):
    """A name without country code is searched in the given country.

    And merged with the institution already at the address found.
    """
    monkeypatch.setattr(
        _institutions_update,
        "_get_location",
        lambda geocode, address, country=None: FakeLocation(GENEVA) if country == "ch" else None,
    )

    cli(["bids_inventory", "institutions", "relocate", "University of Geneva", "--country", "CH"])

    saved = json.loads(mislocated.read_text())
    assert "University of Geneva" not in saved
    campus = saved["Campus Biotech, Chemin des Mines 9,Geneve,District,CH,1211"]
    assert campus["AlternativeNames"] == ["University of Geneva"]


def test_relocate_not_found(mislocated, monkeypatch):
    """An institution not found in its country is not located anymore."""
    monkeypatch.setattr(
        _institutions_update, "_get_location", lambda geocode, address, country=None: None
    )

    cli(["bids_inventory", "institutions", "relocate", "University of Geneva", "--country", "ch"])

    saved = json.loads(mislocated.read_text())
    assert "University of Geneva" in saved[UNLOCATED]["AlternativeNames"]


def test_relocate_unknown_name(mislocated):
    with pytest.raises(SystemExit):
        cli(["bids_inventory", "institutions", "relocate", "Nowhere"])


def test_relocate_far_from_postcode(mislocated, monkeypatch):
    """In the right country but far from its ZIP code: located at its ZIP code.

    Results that are only a country or a region are not used.
    """
    name = "IBMI, Brenneckestr._6_Magdeburg_District_DE_39118"
    braunschweig = {
        **MAGDEBURG,
        "address": "District, Braunschweig",
        "latitude": 52.26,
        "longitude": 10.52,
    }
    monkeypatch.setattr(
        _institutions_update,
        "_get_location",
        lambda geocode, address, country=None: FakeLocation(braunschweig),
    )
    postcodes = []

    def fake_postcode_location(geocode, postcode, country):
        postcodes.append((postcode, country))
        return FakeLocation({**MAGDEBURG, "address": "39118, Magdeburg, Deutschland"})

    monkeypatch.setattr(_institutions_update, "_postcode_location", fake_postcode_location)

    cli(["bids_inventory", "institutions", "relocate", name])

    saved = json.loads(mislocated.read_text())
    assert saved[name]["Address"] == "39118, Magdeburg, Deutschland"
    assert postcodes == [("39118", "de")]


def test_vague_locations_are_not_used(monkeypatch):
    calls = []

    def fake_geocode(address, **options):
        calls.append(address)
        return FakeLocation({**GENEVA, "address": "Shqipëria", "place_rank": 4})

    assert _institutions_update._get_location(fake_geocode, "Somewhere, Birmingham", "al") is None
    # the parts of the address were tried too
    assert calls == ["Somewhere, Birmingham", "Birmingham", "Somewhere"]
