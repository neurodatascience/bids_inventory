"""Tools for basic figures about the known datasets."""

import re
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
DURATION_DATATYPES = ["func", "eeg", "ieeg", "meg", "nirs", "motion", "pet", "emg"]


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
        marginal_y="histogram",
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


# same color for a datatype in all the figures, whatever the datatypes shown
DATATYPE_COLORS = dict(zip(DURATION_DATATYPES, px.colors.qualitative.Plotly))


def _run_durations(runs: Any) -> list[float]:
    """Return the duration of each run of a datatype (in seconds, known ones only)."""
    if isinstance(runs, dict):
        runs = [run for task_runs in runs.values() for run in task_runs or []]
    return [float(run) for run in runs or [] if run is not None and run == run and run > 0]


def run_durations_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the distribution of the duration of all the runs, one histogram per datatype.

    The runs of the first participant of each dataset (the only ones in the listing).
    The bins are logarithmic: runs last from seconds to days.
    """
    minutes = {
        datatype: [
            seconds / 60
            for duration in df["duration"]
            for seconds in _run_durations((duration or {}).get(datatype))
        ]
        for datatype in DURATION_DATATYPES
    }
    all_minutes = [x for values in minutes.values() for x in values]
    fig = go.Figure()
    if not all_minutes:
        return fig.update_layout(title="duration of the runs")
    edges = np.logspace(
        np.floor(np.log10(min(all_minutes)) * 4) / 4,
        np.ceil(np.log10(max(all_minutes)) * 4) / 4,
        num=50,
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    for datatype, values in minutes.items():
        if not values:
            continue
        counts, _ = np.histogram(values, bins=edges)
        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=np.diff(edges),
                name=f"{datatype} ({len(values)} runs)",
                marker_color=DATATYPE_COLORS[datatype],
                opacity=0.6,
                customdata=np.stack([edges[:-1], edges[1:]], axis=-1),
                hovertemplate=(
                    f"{datatype}<br>%{{customdata[0]:.3g}} to %{{customdata[1]:.3g}} minutes"
                    "<br>%{y} runs<extra></extra>"
                ),
            )
        )
    return fig.update_layout(
        barmode="overlay",
        title=("Duration of the runs<br><sup>runs of the first participant of each dataset</sup>"),
        xaxis={"type": "log", "title": {"text": "duration (minutes)"}},
        yaxis={"title": {"text": "number of runs"}},
    )


def _version_key(version: str) -> tuple[Any, ...]:
    """Sort versions by their numbers: 1.10.0 after 1.9.0, 23.2.0 after 0.16.1."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"[.+-]", version)
    )


def version_histogram(df: pd.DataFrame, column: str, name: str) -> go.Figure:
    """Plot the number of datasets for each version (of BIDS, fMRIPrep, MRIQC...).

    ``column``: column of the listing with the versions (``bids_version``...).
    """
    versions = df[column].replace("n/a", None).dropna().astype(str)
    counts = versions.value_counts()
    counts = counts.loc[sorted(counts.index, key=_version_key)]
    # go.Bar and not px.bar: there may be no versions at all
    fig = go.Figure(
        go.Bar(
            x=list(counts.index),
            y=list(counts.values),
            hovertemplate=f"{name} %{{x}}<br>%{{y}} datasets<extra></extra>",
        )
    )
    return fig.update_layout(
        title=f"{name} version ({len(versions)} datasets)",
        xaxis={"type": "category", "title": {"text": f"{name} version"}},
        yaxis={"title": {"text": "number of datasets"}},
    )


# names of the file formats in the listing, by datatype
FILE_FORMATS = {
    "eeg": {"eeg": "BrainVision", "edf": "EDF", "bdf": "BDF", "set": "EEGLAB"},
    "ieeg": {"eeg": "BrainVision", "edf": "EDF", "set": "EEGLAB", "mefd": "MEF3", "nwb": "NWB"},
    "meg": {
        ".fif": "FIF (Elekta/MEGIN)",
        ".ds": "CTF",
        "": "BTi/4D",
        ".con": "KIT/Yokogawa",
        ".kdf": "KRISS",
        ".raw.mhd": "ITAB",
    },
}


def file_formats_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the number of datasets using each file format, for eeg, ieeg and meg."""
    fig = make_subplots(rows=1, cols=len(FILE_FORMATS), subplot_titles=list(FILE_FORMATS))
    for col, (datatype, names) in enumerate(FILE_FORMATS.items(), start=1):
        counts = {name: 0 for name in names.values()}
        for formats in df[f"{datatype}_file_formats"]:
            for fmt, nb_files in (formats if isinstance(formats, dict) else {}).items():
                if nb_files and fmt in names:
                    counts[names[fmt]] += 1
        fig.add_trace(
            go.Bar(
                x=list(counts),
                y=list(counts.values()),
                name=datatype,
                marker_color=DATATYPE_COLORS[datatype],
                hovertemplate="%{x}<br>%{y} datasets<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_yaxes(title_text="number of datasets", row=1, col=1)
    return fig.update_layout(title="File formats of the EEG, iEEG and MEG data", showlegend=False)


CHANNEL_DATATYPES = ["eeg", "ieeg", "meg", "emg"]


def channels_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the distribution of the number of channels, one histogram per datatype.

    One value per dataset: the largest number of channels of the runs of its first participant
    (``nb_channels`` column, from the JSON sidecars).
    """
    fig = make_subplots(rows=1, cols=len(CHANNEL_DATATYPES), subplot_titles=CHANNEL_DATATYPES)
    for col, datatype in enumerate(CHANNEL_DATATYPES, start=1):
        column = df["nb_channels"] if "nb_channels" in df else []
        values = [
            max(channels[datatype])
            for channels in column
            if isinstance(channels, dict) and channels.get(datatype)
        ]
        fig.add_trace(
            go.Histogram(
                x=values,
                name=f"{datatype} ({len(values)} datasets)",
                marker_color=DATATYPE_COLORS[datatype],
                hovertemplate="%{x} channels<br>%{y} datasets<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="number of channels", row=1, col=col)
    fig.update_yaxes(title_text="number of datasets", row=1, col=1)
    return fig.update_layout(
        title=(
            "Number of channels<br><sup>"
            "largest number of channels of the runs of the first participant of each dataset</sup>"
        ),
        showlegend=False,
    )


def nb_datatypes_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the number of datasets for each number of datatypes they have."""
    counts = df["datatypes"].apply(len).value_counts().sort_index()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        text=counts.values,
        labels={"x": "number of datatypes", "y": "number of datasets"},
        title="Number of datatypes per dataset",
    )
    return fig.update_xaxes(dtick=1)


# datasets with physio files whose recording labels and columns are not known
NO_RECORDING_LABEL = "not identified"


def physio_types_histogram(df: pd.DataFrame) -> go.Figure:
    """Plot the number of datasets with physio files of each type of signal.

    From the recording labels of the physio files and the columns of their sidecars
    (``physio_types`` column): a dataset with several types is counted for each one.
    The datasets with physio files whose types are not identified are counted apart.
    """
    with_physio = df[df["has_physio"].fillna(False).astype(bool)]
    types = with_physio["physio_types"].apply(
        lambda x: x if isinstance(x, list) and x else [NO_RECORDING_LABEL]
    )
    counts = types.explode().value_counts()
    # the most common types first, the datasets without labels last
    order = [x for x in counts.index if x != NO_RECORDING_LABEL]
    order += [x for x in counts.index if x == NO_RECORDING_LABEL]
    counts = counts.loc[order]
    fig = go.Figure(
        go.Bar(
            x=list(counts.index),
            y=list(counts.values),
            text=list(counts.values),
            hovertemplate="%{x}<br>%{y} datasets<extra></extra>",
        )
    )
    return fig.update_layout(
        title=(
            f"Types of physiological recordings ({len(with_physio)} datasets with physio files)"
            "<br><sup>from the recording labels of the physio files and the columns "
            "of their sidecars: a dataset can have several types</sup>"
        ),
        xaxis={"type": "category", "title": {"text": "type of recording"}},
        yaxis={"title": {"text": "number of datasets"}},
    )
