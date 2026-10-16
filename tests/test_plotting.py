import math

import pandas as pd
import pytest

pytest.importorskip("plotly")

from bids_inventory._plotting import (  # noqa: E402
    neurobagel_age_distribution,
    neurobagel_bar_per_dataset,
    neurobagel_female_ratio_vs_participants,
    neurobagel_per_dataset,
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
