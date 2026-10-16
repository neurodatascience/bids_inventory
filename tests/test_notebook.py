import pytest

# needs the 'browse' extra
pytest.importorskip("marimo")


def test_browse_app_runs():
    """The dashboard notebook runs, and keeps all the datasets with a datatype by default."""
    from bids_inventory import browse_app

    _, defs = browse_app.app.run()

    df, filtered_df = defs["df"], defs["filtered_df"]
    assert len(filtered_df) == (df["datatypes"].apply(len) > 0).sum()

    # one row per participant of the selected datasets annotated by Neurobagel
    nb_participants = defs["nb_participants"]
    assert set(nb_participants["dataset_id"]) <= set(filtered_df["name"])
    assert not nb_participants.duplicated(subset=["dataset_id", "participant_id"]).any()
    # by default, only the datasets about humans (or without species)
    species = defs["dataset_species"][nb_participants["dataset_id"].unique()]
    assert {x for x in species for x in x} <= {"homo sapiens", "not specified"}


@pytest.mark.parametrize("edit, mode", [(False, "run"), (True, "edit")])
def test_browse_command(monkeypatch, edit, mode):
    """`bids_inventory browse` opens the notebook with marimo."""
    import subprocess

    from bids_inventory import _browse
    from bids_inventory._cli import cli

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))

    cli(["bids_inventory", "browse", *(["--edit"] if edit else [])])

    assert calls[0][-3:] == ["marimo", mode, str(_browse.NOTEBOOK)]
    assert _browse.NOTEBOOK.exists()
