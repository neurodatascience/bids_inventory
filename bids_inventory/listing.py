"""Load and save the listing of known BIDS datasets."""

import functools
import json
from collections.abc import Iterable
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _bids_datatypes_file() -> Path:
    # not _data_dir(): defined further down, and needed when the module is imported
    return Path(__file__).parent / "data" / "bids_datatypes.json"


def bids_datatypes() -> dict[str, str]:
    """Return the BIDS datatypes and their names (from the BIDS schema).

    Saved in bids_inventory/data/bids_datatypes.json by ``bids_inventory update``.
    """
    with open(_bids_datatypes_file()) as f:
        datatypes: dict[str, str] = json.load(f)["datatypes"]
    return datatypes


# the official BIDS datatypes (anat, func, eeg...)
KNOWN_DATATYPES = sorted(bids_datatypes())


@functools.lru_cache(maxsize=1)
def _physio_recordings() -> dict[str, list[str]]:
    """Map the (normalized) recording labels of the physio files to their types."""
    with open(Path(__file__).parent / "data" / "physio_recordings.json") as f:
        content = json.load(f)
    types = {label: [kind] for kind, labels in content["types"].items() for label in labels}
    return {**types, **content["combined"]}


def physio_recording_types(labels: Iterable[str]) -> list[str]:
    """Return the types of signals of physio files from their recording labels.

    For example ["cardiac", "respiratory"] for ["respcardiac"],
    see bids_inventory/data/physio_recordings.json ("other" for unknown labels).
    """
    known = _physio_recordings()
    types = set()
    for label in labels:
        # ECG2 is ecg, eye1 is eye
        types.update(known.get(str(label).lower().rstrip("0123456789"), ["other"]))
    return sorted(types)


@functools.lru_cache(maxsize=1)
def known_datasets_df() -> pd.DataFrame:
    """Return dataframe of all datasets in the inventory.

    Returns
    -------
    pd.DataFrame
        Dataframe containing list of datasets known to the inventory.

    A data dictionary can be found in:
    ``bids_inventory/data/columns_description.json``
    """
    openneuro_df = _load_known_datasets(_openneuro_listing())
    non_opnenneuro_df = _load_known_datasets(_non_openneuro_listing())
    return pd.concat([openneuro_df, non_opnenneuro_df])


def neurobagel_participants_df() -> pd.DataFrame:
    """Return the participants of the datasets annotated by Neurobagel.

    One row per participant (and session) of each dataset of
    https://github.com/OpenNeuroDatasets-JSONLD annotated by Neurobagel,
    with the age (in years) and the sex (female, male, other)
    standardized across datasets.

    Updated with ``bids_inventory neurobagel``.
    """
    return pd.read_csv(
        _data_dir() / "neurobagel_participants.tsv",
        sep="\t",
        dtype={"dataset_id": str, "participant_id": str, "session_id": str, "sex": str},
    )


def neurobagel_datasets_df() -> pd.DataFrame:
    """Return the repositories of https://github.com/OpenNeuroDatasets-JSONLD.

    With why they were used or not in ``neurobagel_participants_df``,
    and the species of their participants (empty if not specified).

    Updated with ``bids_inventory neurobagel``.
    """
    return pd.read_csv(
        _data_dir() / "neurobagel_datasets.tsv", sep="\t", dtype=str, keep_default_na=False
    )


def dataset_species() -> dict[str, list[str]]:
    """Return the species of the participants of the datasets, when they are known.

    From the species column of the participants.tsv of the datasets annotated by Neurobagel
    (``neurobagel_datasets_df``), and from bids_inventory/data/species.tsv
    for datasets without one. Most datasets are about humans but do not say so:
    they are not listed.
    """
    datasets = neurobagel_datasets_df()
    species = dict(zip(datasets["dataset_id"], datasets["species"]))
    listed = pd.read_csv(_species_file(), sep="\t", dtype=str, keep_default_na=False)
    species.update(zip(listed["dataset_id"], listed["species"]))
    return {name: x.split(", ") for name, x in species.items() if x}


def _species_file() -> Path:
    return _data_dir() / "species.tsv"


def _data_dir() -> Path:
    return Path(__file__).parent / "data"


def _openneuro_listing() -> Path:
    return _data_dir() / "openneuro.jsonl"


def _non_openneuro_listing() -> Path:
    return _data_dir() / "non_openneuro.jsonl"


def _skipped_datasets_file() -> Path:
    return _data_dir() / "skipped_datasets.csv"


def skipped_datasets() -> dict[str, str]:
    """Return the datasets that must not be indexed, with the reason why.

    For example datasets deleted from OpenNeuro but still mirrored on GitHub.
    """
    df = pd.read_csv(_skipped_datasets_file(), dtype=str, keep_default_na=False)
    return dict(zip(df["name"], df["reason"]))


def remove_datasets(names: Iterable[str], listing: Path) -> list[str]:
    """Remove datasets from a listing, for example so that the next update indexes them again.

    The other entries are kept as they are.
    If a re-index is in progress (``<listing>.reindexed``),
    the datasets are also removed from it so that it indexes them again.

    Returns
    -------
    The names of the datasets that were removed.
    """
    to_remove = set(names)
    lines = listing.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)["name"] not in to_remove]
    removed = sorted({json.loads(line)["name"] for line in lines} & to_remove)
    if removed:
        tmp_file = listing.with_name(f"{listing.name}.tmp")
        tmp_file.write_text("".join(kept), encoding="utf-8")
        tmp_file.replace(listing)

    reindex_file = listing.with_name(f"{listing.name}.reindexed")
    if reindex_file.exists():
        done = reindex_file.read_text().splitlines()
        reindex_file.write_text("".join(f"{x}\n" for x in done if x.strip() not in to_remove))

    return removed


# columns of integers that can be missing
NULLABLE_INT_COLUMNS = ["readme_size", "bidsignore_size"]


def _load_known_datasets(listing: Path) -> pd.DataFrame:
    """Load a listing of datasets stored as JSON lines (one dataset per line)."""
    df = pd.read_json(listing, lines=True, dtype=False, convert_dates=False)
    df["created_on"] = pd.to_datetime(df["created_on"], utc=True)
    # integers with missing values: not floats
    for column in NULLABLE_INT_COLUMNS:
        if column in df:
            df[column] = df[column].astype("Int64")
    return df


def save_known_datasets(df: pd.DataFrame, output_file: Path) -> None:
    """Save a listing of datasets as JSON lines (one dataset per line, sorted by name).

    Missing values (``NaN``, ``None``, ``"n/a"``) are saved as ``null``
    and durations are saved in seconds.
    """
    df = df.copy()
    df["duration"] = df["duration"].apply(_convert_duration)
    # dates can come in different formats (from the listing or from git):
    # parse them one by one to avoid pandas guessing a single format
    df["created_on"] = df["created_on"].apply(
        lambda x: pd.to_datetime(x, utc=True, errors="coerce")
    )
    records = sorted(df.to_dict(orient="records"), key=lambda x: str(x["name"]))
    # write to a temporary file first so that an interrupted save
    # cannot leave a truncated listing behind
    tmp_file = output_file.with_name(f"{output_file.name}.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for record in records:
            record = {str(k): _to_json_value(v) for k, v in record.items()}
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    tmp_file.replace(output_file)


def _to_json_value(value: Any) -> Any:
    """Convert a value to something that can be saved as JSON."""
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_json_value(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NaT or value is pd.NA or value == "n/a":
        return None
    if isinstance(value, float) and isnan(value):
        return None
    return value


# in seconds: longer func and pet runs had their duration read in milliseconds
MAX_RUN_DURATION = 10 * 3600


def _run_duration(run: Any) -> float | None:
    """Return the duration of a run in seconds.

    A run can already be a duration in seconds,
    or a pair ``(number of samples, duration of a sample)``
    (for example ``(number of volumes, repetition time)``).
    Returns ``None`` if the duration is unknown.
    """
    if run is None:
        return None
    if isinstance(run, (int, float)):
        duration = float(run)
    elif any(i is None or isnan(i) for i in run):
        return None
    else:
        duration = float(np.prod(run))
    # a duration of 0 means the duration could not be read
    return None if isnan(duration) or duration <= 0 else duration


def _convert_duration(
    durations: dict[str, Any],
) -> dict[str, list[float] | dict[str, list[float] | None]]:
    """Convert the duration of all runs to seconds.

    ``durations`` is either ``{datatype: [run, ...]}``
    or ``{datatype: {task: [run, ...]}}``.
    Runs with an unknown duration are dropped from the task lists
    and tasks without any known duration are set to ``None``.

    func and pet runs of more than 10 hours had their duration read in milliseconds
    (for example 180 volumes of 2000 "seconds", or a FrameDuration in milliseconds):
    they are converted to seconds.
    """
    if not isinstance(durations, dict) or not durations:
        return {}

    converted: dict[str, list[float] | dict[str, list[float] | None]] = {}
    for datatype, value in durations.items():

        def duration(run: Any) -> float | None:
            d = _run_duration(run)
            if datatype in {"func", "pet"} and d is not None and d > MAX_RUN_DURATION:
                d /= 1000
            return d

        if isinstance(value, list):
            converted[datatype] = [np.nan if (d := duration(run)) is None else d for run in value]
        elif isinstance(value, dict):
            converted[datatype] = {
                task: [d for run in runs or [] if (d := duration(run)) is not None] or None
                for task, runs in value.items()
            }

    return converted


def is_known_dataset(dataset_name: str) -> bool:
    """Check if a dataset is in the inventory.

    Parameters
    ----------
    dataset_name : :obj:`str`
        Name of the dataset to check, for example ``ds000117``.
    """
    openneuro = known_datasets_df()
    mask = openneuro.name == dataset_name
    return mask.sum() != 0
