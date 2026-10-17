import math

import pandas as pd
import pytest

pytest.importorskip("plotly")

from bids_inventory._plotting import (  # noqa: E402
    hours_per_datatype,
    neurobagel_age_distribution,
    neurobagel_bar_per_dataset,
    neurobagel_female_ratio_vs_participants,
    neurobagel_per_dataset,
    plot_hours_vs_time,
)


@pytest.fixture
def participants():
    return pd.DataFrame(
        {
            "dataset_id": ["ds000001"] * 4 + ["ds000002"] * 2,
            # sub-01 has 2 sessions in ds000001
            "participant_id": ["sub-01", "sub-01", "sub-02", "sub-03", "sub-01", "sub-02"],
            "age": [20.0, 21.0, None, 30.0, 40.0, 50.0],
            "sex": ["female", "female", "male", None, "male", "male"],
        }
    )


def test_neurobagel_per_dataset(participants):
    per_dataset = neurobagel_per_dataset(participants).set_index("dataset_id")

    assert per_dataset["nb_participants"].tolist() == [3, 2]
    assert per_dataset["missing_age"].tolist() == pytest.approx([100 / 3, 0])
    assert per_dataset["missing_sex"].tolist() == pytest.approx([100 / 3, 0])
    # among the participants with a known sex
    assert per_dataset["female_ratio"].tolist() == [0.5, 0]


def test_neurobagel_plots(participants):
    per_dataset = neurobagel_per_dataset(participants)

    # datasets without missing values are not shown
    assert list(neurobagel_bar_per_dataset(per_dataset, "missing_age").data[0].x) == ["ds000001"]
    assert len(neurobagel_female_ratio_vs_participants(per_dataset).data[0].x) == 2

    fig = neurobagel_age_distribution(participants)
    assert [trace.name for trace in fig.data] == ["female", "male"]
    # one row per participant, with age and sex
    assert sorted(fig.data[1].x) == [40, 50]
    assert not any(math.isnan(x) for x in fig.data[0].x)


def test_hours_per_datatype():
    df = pd.DataFrame(
        {
            "name": ["ds000001", "ds000002", "ds000003"],
            "created_on": pd.to_datetime(["2020-01-01", "2019-01-01", "2021-01-01"]),
            "nb_subjects": [10, 2, 5],
            "duration": [
                # per task, one task without known duration
                {"func": {"a": [360.0, 360.0], "b": None}},
                {"eeg": [1800.0, float("nan"), None], "func": {"a": [720.0]}},
                {"pet": [None]},
            ],
        }
    )
    hours = hours_per_datatype(df)
    assert hours[["name", "datatype"]].values.tolist() == [
        ["ds000001", "func"],
        ["ds000002", "func"],
        ["ds000002", "eeg"],
    ]
    # duration of the first participant times the number of participants
    assert hours["hours"].tolist() == pytest.approx([2, 0.4, 1])

    fig = plot_hours_vs_time(df)
    assert [trace.name.split(":")[0] for trace in fig.data] == ["func", "eeg"]
    # cumulative, by creation date
    assert list(fig.data[0].y) == pytest.approx([0.4, 2.4])


def test_field_strength_histogram():
    from bids_inventory._plotting import field_strength_histogram

    df = pd.DataFrame(
        {
            "name": ["ds1", "ds2", "ds3", "ds4"],
            "nb_subjects": [10, 20, 30, 40],
            "field_strengths": [["3T", "7T"], ["unknown"], ["10.5T"], []],
        }
    )
    fig = field_strength_histogram(df)
    # sorted by field strength, unknown last; datasets without MRI not counted
    assert list(fig.data[0].x) == ["3T", "7T", "10.5T", "unknown"]
    assert list(fig.data[0].y) == [1, 1, 1, 1]
