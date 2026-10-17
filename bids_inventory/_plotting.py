"""Tools for basic figures about the known datasets."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from bids_inventory.listing import KNOWN_DATATYPES

LABELS = {
    "nb_subjects": "number of participants",
    "nb_sessions": "number of sessions",
    "mean_size": "data per participants (bytes)",
    "nb_tasks": "number of tasks",
    "nb_datatypes": "number of datatypes",
    "total_duration": "'scan time' per participants (hours)",
}


def scatter_subject_vs(
    df: pd.DataFrame,
    y: str,
    size: None | str = None,
    color: None | str = "is_openneuro",
    title: None | str = "BIDS open datasets",
    log_x: bool = True,
    log_y: bool = True,
    marginal: None | str | bool = "box",
) -> go.Figure:
    return px.scatter(
        df,
        x="nb_subjects",
        y=y,
        size=size,
        log_x=log_x,
        log_y=log_y,
        title=title,
        hover_name="name",
        marginal_x=marginal,
        marginal_y=marginal,
        labels=LABELS,
        color=color,
    )


def histogram_tasks(df: pd.DataFrame) -> go.Figure:
    tasks = []
    for x in df.tasks:
        tasks.extend(x)

    new_df = pd.DataFrame({"tasks": tasks})

    count = new_df.tasks.value_counts()
    order = [i[0] for i in count.items()]
    return px.histogram(
        new_df, x="tasks", category_orders=dict(tasks=order), title="tasks distribution"
    )


def plot_dataset_size_vs_time(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values(by=["created_on"])

    y = []
    for col in ["size", "nb_subjects"]:
        cumsum = df[col].cumsum()
        col = f"cumsum_{col}"
        df[col] = cumsum
        y.append(col)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for col in y:
        secondary_y = False
        if col == "cumsum_nb_subjects":
            secondary_y = True

        fig.add_trace(
            go.Scatter(
                name=col.replace("cumsum_", ""), x=df["created_on"], y=df[col], mode="lines"
            ),
            secondary_y=secondary_y,
        )
    fig.update_layout(title="Cumulative BIDS data across time", hovermode="x")
    fig.update_yaxes(title_text="data size [bytes]", secondary_y=False, nticks=10)
    fig.update_yaxes(title_text="number of subjects", secondary_y=True, nticks=10)
    fig.update_xaxes(title_text="time")

    return fig


# datatypes whose duration is in the listing
DURATION_DATATYPES = ["func", "eeg", "ieeg", "meg", "nirs", "motion", "pet"]


def _runs_duration(runs: Any) -> float:
    """Return the total duration of the runs of a datatype (in seconds, NaN if unknown)."""
    if isinstance(runs, dict):
        runs = [run for task_runs in runs.values() for run in task_runs or []]
    # unknown durations are NaN, or null in the listing
    runs = [run for run in runs or [] if run is not None and run == run]
    return float(sum(runs)) if runs else np.nan


def hours_per_datatype(df: pd.DataFrame) -> pd.DataFrame:
    """Return the number of hours of data of each dataset, for each datatype with durations.

    Estimated as the duration of the runs of the first participant
    times the number of participants
    (only the runs of the first participant are in the listing).

    One row per dataset and datatype with a known duration:
    ``name``, ``created_on``, ``datatype``, ``hours``.
    """
    rows = []
    for _, dataset in df.iterrows():
        for datatype in DURATION_DATATYPES:
            seconds = _runs_duration((dataset["duration"] or {}).get(datatype))
            if seconds == seconds and seconds > 0:
                rows.append(
                    {
                        "name": dataset["name"],
                        "created_on": dataset["created_on"],
                        "datatype": datatype,
                        "hours": seconds * dataset["nb_subjects"] / 3600,
                    }
                )
    return pd.DataFrame(rows, columns=["name", "created_on", "datatype", "hours"])


def plot_hours_vs_time(df: pd.DataFrame, log_y: bool = True) -> go.Figure:
    """Plot the cumulative number of hours of data for each datatype, by creation date.

    See ``hours_per_datatype`` for how the hours are estimated.
    """
    hours = hours_per_datatype(df).sort_values("created_on")
    fig = go.Figure()
    for datatype in DURATION_DATATYPES:
        datatype_hours = hours[hours["datatype"] == datatype]
        if datatype_hours.empty:
            continue
        fig.add_trace(
            go.Scatter(
                name=(
                    f"{datatype}: {datatype_hours['hours'].sum():,.0f} hours "
                    f"({len(datatype_hours)} datasets)"
                ),
                x=datatype_hours["created_on"],
                y=datatype_hours["hours"].cumsum(),
                text=datatype_hours["name"],
                mode="lines",
                line_shape="hv",
                hovertemplate="%{text}<br>%{x}<br>%{y:,.0f} hours",
            )
        )
    fig.update_layout(
        title=(
            "Hours of data, by creation date of the datasets<br><sup>"
            "estimated from the runs of the first participant times the number of participants"
            " (durations only known for "
            f"{', '.join(DURATION_DATATYPES)})</sup>"
        ),
        xaxis_title={"text": "date"},
        yaxis_title={"text": "cumulative hours"},
        yaxis_type="log" if log_y else "linear",
    )
    return fig


def datatypes_histogram(df: pd.DataFrame) -> go.Figure:
    datatypes_df = df[KNOWN_DATATYPES].sum()
    fig = px.bar(
        datatypes_df,
        labels={"index": "datatype", "value": "count"},
        hover_name=None,
        hover_data=None,
        title="datatypes in datasets",
    )
    return fig


def field_strength_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the number of datasets (and participants) for each MRI field strength.

    ``df`` needs the ``field_strengths`` column added by ``wrangle_data``.
    A dataset with data at several field strengths is counted for each one.
    """
    rows = (
        df[["name", "nb_subjects", "field_strengths"]]
        .explode("field_strengths")
        .dropna(subset=["field_strengths"])
    )
    counts = rows.groupby("field_strengths").agg(
        datasets=("name", "nunique"), participants=("nb_subjects", "sum")
    )

    def order(label: str) -> float:
        return float(label.removesuffix("T")) if label.endswith("T") else float("inf")

    counts = counts.loc[sorted(counts.index, key=order)].reset_index()
    return px.bar(
        counts,
        x="field_strengths",
        y="datasets",
        hover_data=["participants"],
        text="datasets",
        labels={"field_strengths": "MRI field strength", "datasets": "number of datasets"},
        title="MRI field strength of the datasets",
    ).update_xaxes(type="category")


def institutions_map(locations: pd.DataFrame) -> go.Figure:
    """Plot the located institutions on a world map.

    ``locations``: as returned by ``bids_inventory.institutions.institutions_locations``.
    The size of each marker is the number of datasets of the institution.
    """
    located = locations.dropna(subset=["latitude", "longitude"])
    return px.scatter_geo(
        located,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data=["address", "count", "nb_subjects"],
        projection="natural earth",
        size="count",
        title="institutions",
    )


NEUROBAGEL_LABELS = {
    "dataset_id": "dataset",
    "nb_participants": "number of participants",
    "missing_age": "participants with missing age (%)",
    "missing_sex": "participants with missing sex (%)",
    "female_ratio": "proportion of females",
}


def neurobagel_unique_participants(participants: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per participant of each dataset (the first one, for example its first session).

    ``participants``: as returned by ``bids_inventory.listing.neurobagel_participants_df``.
    """
    return participants.drop_duplicates(subset=["dataset_id", "participant_id"])


def neurobagel_per_dataset(participants: pd.DataFrame) -> pd.DataFrame:
    """Return statistics about the participants of each dataset.

    Their number, the percentage of them with missing age and sex,
    and the proportion of females among those with a known sex.
    """
    participants = neurobagel_unique_participants(participants).assign(
        missing_age=lambda x: x["age"].isna() * 100,
        missing_sex=lambda x: x["sex"].isna() * 100,
        female=lambda x: (x["sex"] == "female").where(x["sex"].notna()),
    )
    return participants.groupby("dataset_id", as_index=False).agg(
        nb_participants=("participant_id", "count"),
        missing_age=("missing_age", "mean"),
        missing_sex=("missing_sex", "mean"),
        female_ratio=("female", "mean"),
    )


def neurobagel_bar_per_dataset(per_dataset: pd.DataFrame, column: str) -> go.Figure:
    """Bar plot of a column of ``neurobagel_per_dataset``, sorted, without the zeros."""
    per_dataset = per_dataset[per_dataset[column].fillna(0) != 0].sort_values(by=column)
    return px.bar(
        per_dataset,
        x="dataset_id",
        y=column,
        hover_data=["nb_participants"],
        labels=NEUROBAGEL_LABELS,
        title=f"{NEUROBAGEL_LABELS[column]} for each dataset",
    )


def neurobagel_female_ratio_vs_participants(per_dataset: pd.DataFrame) -> go.Figure:
    return px.scatter(
        per_dataset.dropna(subset=["female_ratio"]),
        x="nb_participants",
        y="female_ratio",
        log_x=True,
        hover_name="dataset_id",
        marginal_x="box",
        marginal_y="box",
        labels=NEUROBAGEL_LABELS,
        title="proportion of females VS number of participants for each dataset",
    )


def neurobagel_age_distribution(participants: pd.DataFrame) -> go.Figure:
    """Histograms of the age of the participants, for each sex."""
    participants = neurobagel_unique_participants(participants).dropna(subset=["age", "sex"])
    fig = go.Figure()
    for sex in ["female", "male", "other"]:
        ages = participants.loc[participants["sex"] == sex, "age"]
        if len(ages):
            fig.add_trace(go.Histogram(x=ages.round(), name=sex, xbins={"size": 1}))
    fig.update_layout(
        barmode="overlay",
        title=(
            f"Age distribution ({len(participants)} participants "
            f"of {participants['dataset_id'].nunique()} datasets)"
        ),
        xaxis_title={"text": "age (years)"},
        yaxis_title={"text": "count"},
    )
    fig.update_traces(opacity=0.5)
    return fig
