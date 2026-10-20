"""Add summary columns to the listing of datasets and filter it."""

import re
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from bids_inventory.institutions import dataset_institutions, known_names, load_institutions
from bids_inventory.listing import (
    KNOWN_DATATYPES,
    dataset_species,
    neurobagel_participants_df,
    physio_column_types,
    physio_recording_types,
)
from bids_inventory.logger import bi_logger

cc_log = bi_logger()


def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    """Do general wrangling of the known datasets.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of known datasets

    Returns
    -------
    df : pd.DataFrame
        dataframe of known datasets with extra columns

    - ``nb_datatypes``: :obj:`int` Number of unique BIDS supported datatypes in the dataset.
    - ``nb_sessions``: :obj:`int` Total number of unique sessions in the dataset.
    - ``nb_authors``: :obj:`int` Number of authors as reported in the description of the dataset.
    - ``nb_tasks``: :obj:`int` Total number of unique tasks in the dataset.
    - ``useful_participants_tsv``: :obj:`bool`
    - ``has_physio``: :obj:`bool` ``True`` if the dataset contains any ``*_physio.tsv.gz`` files.
    - ``physio_types``: :obj:`list[str]` Types of signals of its physio files (cardiac,
      respiratory, eye tracking...), from their recording labels and the columns of their sidecars.
    - ``has_fmriprep``: :obj:`bool`` ``True`` if the dataset has knwow fmriprep preprocessed derivatives.
    - ``has_freesurfer``: :obj:`bool` ``True`` if the dataset has knwow freesurfer preprocessed derivatives.
    - ``has_mriqc``: :obj:`bool` ``True`` if the dataset has knwow mriqc derivatives.
    - ``source``: Specifies the source of the dataset.
    - ``size``: size in bytes, without the derivatives and sourcedata folders if known
      (``raw_size``), otherwise of the whole dataset (as reported by datalad).
    - ``mean_size``: size per subject in bytes
    - datatype: one column for each BIDS known datatype with ``True`` if this dataset contains that datatype
      (``phenotype``: if it has a phenotype folder at its root, ``has_phenotype_dir``).
    - ``total_duration``: Total "scanned" duration per participant (combines runs from: ``func``, ``eeg``, ``ieeg``)
    - ``institution_names``: :obj:`list[str]` Institutions of the dataset,
      as named in the keys of ``bids_inventory/data/institutions.json``.
    - ``countries``: :obj:`list[str]` Countries of these institutions.
    - ``ages``: :obj:`list[float]` Ages of its participants (in years),
      from neurobagel_participants.tsv (empty if not known).
    - ``field_strengths``: :obj:`list[str]` Nominal field strengths of the MRI scanners
      (``["3T"]``...), ``["unknown"]`` for MRI datasets without MagneticFieldStrength,
      empty for datasets without MRI data.
    - ``species``: :obj:`list[str]` Species of the participants (for example ``["mus musculus"]``),
      empty if not specified: see :func:`bids_inventory.listing.dataset_species`.
    """
    # do not modify the input (for example the cached output of known_datasets_df)
    df = df.copy()
    _detect_duplicate_datasets(df)

    # the phenotype datatype is not in the sub-* folders, but in a folder at the root
    df["datatypes"] = [
        sorted({*datatypes, "phenotype"}) if has_phenotype is True else datatypes
        for datatypes, has_phenotype in zip(df["datatypes"], df["has_phenotype_dir"])
    ]

    df["nb_sessions"] = df["sessions"].apply(lambda x: max(len(x), 1))

    df["nb_datatypes"] = df["datatypes"].apply(lambda x: len(x))

    # if only one column we assume it is only a participant_id file
    useful_participants_tsv = [(len(row[1]["participant_columns"]) > 1) for row in df.iterrows()]
    df["useful_participants_tsv"] = useful_participants_tsv

    # set non empty fmriprep / freesurfer / mriqc to true
    for der in [
        "fmriprep",
        "freesurfer",
        "mriqc",
    ]:
        df[der] = df[der].fillna(False)
        df[f"has_{der}"] = df[der].apply(lambda x: bool(x))

    df["nb_tasks"] = df["tasks"].apply(lambda x: len(x))
    # unknown for the datasets indexed before it was tracked
    if "nb_runs" not in df:
        df["nb_runs"] = pd.NA

    df["source"] = _get_source_study(df)

    # in bytes: without the derivatives and sourcedata folders when known (raw_size),
    # otherwise the size of the whole dataset reported by datalad
    total_size = df["size"].apply(_size_in_bytes)
    raw_size = (
        pd.to_numeric(df["raw_size"], errors="coerce")
        if "raw_size" in df
        else pd.Series(np.nan, index=df.index)
    )
    df["size"] = raw_size.where(raw_size.notna(), total_size).astype(float)
    df["mean_size"] = df["size"] / df["nb_subjects"]

    for datatype in KNOWN_DATATYPES:
        df[datatype] = df["datatypes"].apply(lambda x: datatype in x)

    _detect_missing_duration(df)

    df["nb_authors"] = df["authors"].apply(lambda x: len(x))

    df["has_physio"] = df["nb_physio_files"] > 0
    # from the recording labels of the physio files, and the columns of their sidecars
    no_values = [[]] * len(df)
    df["physio_types"] = [
        sorted(
            {
                *physio_recording_types(labels if isinstance(labels, list) else []),
                *physio_column_types(columns if isinstance(columns, list) else []),
            }
        )
        for labels, columns in zip(
            df["physio_recordings"] if "physio_recordings" in df else no_values,
            df["physio_columns"] if "physio_columns" in df else no_values,
        )
    ]

    df["total_duration"] = df.apply(_compute_total_duration, axis=1)

    species = dataset_species()
    df["species"] = df["name"].apply(lambda x: species.get(x, []))

    df["field_strengths"] = df.apply(_field_strengths, axis=1)

    df["ages"] = _participant_ages(df["name"])

    institutions = load_institutions()
    known = known_names(institutions)
    located = [dataset_institutions(institutions, x, known) for x in df["institutions"]]
    df["institution_names"] = [keys for keys, _ in located]
    df["countries"] = [countries for _, countries in located]

    # new_cols = {
    #     "author_male": [],
    #     "author_female": [],
    #     "author_andy": [],
    #     "author_unknown": [],
    #     "author_mostly_male": [],
    #     "author_mostly_female": [],
    # }
    # for row in df.iterrows():
    #     if len(row[1]["authors"]) == 0:
    #         for key in new_cols:
    #             new_cols[key].append(np.nan)
    #     else:
    #         print(row[1]["name"])
    #         results = {
    #             "male": 0,
    #             "female": 0,
    #             "andy": 0,
    #             "unknown": 0,
    #             "mostly_male": 0,
    #             "mostly_female": 0,
    #         }
    #         total = 0
    #         for author in row[1]["authors"]:
    #             d = gender.Detector()
    #             #  TODO assuming the surname comes first
    #             guess = d.get_gender(author.replace(", ", " ").split(" ")[0])
    #             # print(f"{author}: {guess}")
    #             results[guess] += 1
    #             total += 1
    #         for key in results:
    #             new_cols[f"author_{key}"].append(results[key] / total)

    return df


def _missing_duration(row: pd.Series) -> None | bool:
    if not set(row["datatypes"]).intersection(
        ("pet", "eeg", "ieeg", "meg", "motion", "nirs", "func", "emg")
    ):
        return None
    for _, value in row["duration"].items():
        if isinstance(value, list) and len(value) == 0:
            return True
        if isinstance(value, dict):
            for _, tmp in value.items():
                if not tmp or len(tmp) == 0:
                    return True
    return False


def _compute_total_duration(row: pd.Series) -> float:
    duration = []
    for _, value in row["duration"].items():
        if isinstance(value, list):
            duration.extend(value)
        if isinstance(value, dict):
            for _, tmp in value.items():
                if tmp:
                    duration.extend(tmp)
    # runs whose duration is unknown are null in the listing
    duration = [x for x in duration if x is not None]
    return np.cumsum(duration)[-1] / 3600 if duration else np.nan


def _detect_missing_duration(df: pd.DataFrame) -> None:
    # pandas-stubs has no overload for a row-wise function returning an optional value
    df["missing_duration"] = df.apply(_missing_duration, axis=1)  # pyrefly: ignore[no-matching-overload]
    cc_log.debug(
        f"\nMissing some duration for {df['missing_duration'].sum()} datasets "
        f"({df['missing_duration'].sum() / len(df) * 100:0.1f} %)."
    )
    for datatype in ["func", "ieeg", "eeg", "pet"]:
        mask = df[datatype]
        tmp_df = df[mask]
        if tmp_df.empty:
            continue
        cc_log.debug(
            f"Missing some duration for {tmp_df['missing_duration'].sum()} of {datatype} datasets "
            f"({tmp_df['missing_duration'].sum() / len(tmp_df) * 100:0.1f} %)."
        )


def _detect_duplicate_datasets(df: pd.DataFrame) -> None:
    cc_log.debug("\nDuplicated datasets.")
    tmp = df.name.value_counts()
    cc_log.debug(tmp[tmp > 1])


# source of the datasets, from the beginning of their name
# (ABIDE2 before ABIDE: the first matching prefix is used)
SOURCES = [
    ("ds", "openneuro"),
    ("ABIDE2", "abide 2"),
    ("ABIDE", "abide"),
    ("ADHD200", "adhd200"),
    ("CORR", "corr"),
    ("CNEUROMOD", "neuromod"),
    ("INDI", "indi"),
    ("LABS", "datalad labs"),
    ("STUDYFORREST", "studyforrest"),
    ("DANDI", "dandi"),
    ("NEMAR", "nemar"),
]


def _get_source_study(df: pd.DataFrame) -> list[str]:
    """Return the source of each dataset ("other" for unknown names)."""
    return [
        next((source for prefix, source in SOURCES if name.startswith(prefix)), "other")
        for name in df["name"]
    ]


class FilterConfig(TypedDict):
    fmriprep: bool | None
    mriqc: bool | None
    physio: bool | None
    participants: bool | None
    task: str
    task_regex: bool
    institution: str
    institution_regex: bool
    country: str
    country_regex: bool
    author: str
    author_regex: bool
    datatypes: list[str]
    datatypes_and_or: str
    sources: list[str] | None
    sources_and_or: str
    species: list[str] | None
    field_strength: str | None
    age: list[float] | None
    age_unknown: bool
    sessions: list[int] | None
    runs: list[int] | None


DEFAULT_CONFIG: FilterConfig = {
    "fmriprep": None,
    "mriqc": None,
    "physio": None,
    "participants": None,
    "task": "",
    "task_regex": False,
    "institution": "",
    "institution_regex": False,
    "country": "",
    "country_regex": False,
    "author": "",
    "author_regex": False,
    "datatypes": KNOWN_DATATYPES,
    "datatypes_and_or": "OR",
    "sources": None,
    "sources_and_or": "OR",
    "species": None,
    "field_strength": None,
    "age": None,
    "age_unknown": True,
    "sessions": None,
    "runs": None,
}

# in the species filter: the datasets whose species is not specified (most are about humans)
NOT_SPECIFIED = "not specified"

# datatypes acquired with an MRI scanner
MRI_DATATYPES = {"anat", "func", "dwi", "fmap", "perf"}
# field strength of MRI datasets without MagneticFieldStrength in their sidecars
UNKNOWN_FIELD_STRENGTH = "unknown"
# nominal field strengths (in Tesla) of MRI scanners:
# scanners report values close to them (2.89 for 3T, 6.98 for 7T...)
NOMINAL_FIELD_STRENGTHS = [
    0.064,
    0.35,
    0.5,
    1.0,
    1.5,
    3.0,
    4.0,
    4.7,
    5.0,
    7.0,
    9.4,
    10.5,
    11.7,
    14.0,
    15.2,
    17.2,
]


def field_strength_label(value: float) -> str | None:
    """Return the nominal field strength of a MagneticFieldStrength value ("3T"...).

    Values over 100 are in Gauss (15000 is 1.5T), values under 0.05 are not valid.
    """
    if value > 100:
        value /= 10_000
    if value < 0.05:
        return None
    nominal = min(NOMINAL_FIELD_STRENGTHS, key=lambda x: abs(x - value))
    if abs(nominal - value) <= 0.05 * nominal:
        value = nominal
    return f"{value:g}T"


# units of the sizes reported by datalad: binary (2.2 GB is 2.2 * 1024**3 bytes)
_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4, "PB": 1024**5}


def _size_in_bytes(size: Any) -> float:
    """Return a size reported by datalad ("2.2 GB") in bytes (NaN if not known)."""
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        return float(size)
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGTP]?B)\s*", str(size))
    if not match:
        return np.nan
    return float(match.group(1)) * _SIZE_UNITS[match.group(2)]


def _participant_ages(names: pd.Series) -> list[list[float]]:
    """Return the ages of the participants of each dataset (empty if not known).

    From neurobagel_participants.tsv: one age per participant (the first one if there
    are several sessions), in years (negative ages are left out: not in years).
    """
    participants = neurobagel_participants_df()
    participants = participants[participants["age"] >= 0].drop_duplicates(
        subset=["dataset_id", "participant_id"]
    )
    ages = participants.groupby("dataset_id")["age"].apply(lambda x: sorted(x.tolist()))
    return [ages.get(name, []) for name in names]


def _field_strengths(row: pd.Series) -> list[str]:
    """Return the field strengths of a dataset ("unknown" for MRI data without one)."""
    values = row["field_strength"] if isinstance(row["field_strength"], list) else []
    labels = {label for value in values if (label := field_strength_label(float(value)))}
    if not labels and set(row["datatypes"]) & MRI_DATATYPES:
        return [UNKNOWN_FIELD_STRENGTH]
    return sorted(labels, key=lambda x: float(x.removesuffix("T")))


# text filters: (key of the configuration, column of lists of strings to search)
TEXT_FILTERS = {
    "task": "tasks",
    "institution": "institution_names",
    "country": "countries",
    "author": "authors",
}


def _text_mask(values: pd.Series, query: str, regex: bool) -> pd.Series:
    """Keep the rows where one of the strings of the lists in ``values`` matches ``query``.

    Case insensitive: ``query`` is a substring, or a regular expression if ``regex``.
    Raises ``ValueError`` if the regular expression is invalid.
    """
    if regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regular expression {query!r}: {e}") from e

        def match(x: str) -> bool:
            return pattern.search(x) is not None
    else:

        def match(x: str) -> bool:
            return query.lower() in x.lower()

    return values.apply(
        lambda x: any(match(str(v)) for v in x) if isinstance(x, (list, tuple)) else False
    )


def filter_data(df: pd.DataFrame, config: Any = None) -> pd.DataFrame:
    """Filter the listing of datasets based on some configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Listing of datasets to filter.
    config : Any, default=None
        Should be a :obj:`dict` with any of the following keys.

    - ``"fmriprep"`` : None | bool
    - ``"mriqc"`` : None | bool
    - ``"physio"`` : None | bool
    - ``"task"``: str: keeps the datasets with a task containing this text (case insensitive)
    - ``"task_regex"``: bool: ``"task"`` is a regular expression
    - ``"institution"``: str: same for the institutions
      (as named in the keys of ``bids_inventory/data/institutions.json``)
    - ``"institution_regex"``: bool
    - ``"country"``: str: same for the countries of the institutions
    - ``"country_regex"``: bool
    - ``"author"``: str: same for the authors
    - ``"author_regex"``: bool
    - ``"datatypes"`` : list[str] of any of the BIDS datatypes
    - ``"datatypes_and_or"`` : "OR" | "AND" if any or all of the datatypes must be present
    - ``"sources"`` : list[str] source of the dataset (openneuro, abide...)
    - ``"sources_and_or"`` : "OR" | "AND" if any or all of the sources must be present
    - ``"species"`` : list[str] | None species of the participants (``"homo sapiens"``...),
      ``"not specified"`` for the datasets whose species is not known:
      keeps the datasets whose species are all in the list
    - ``"field_strength"`` : str | None nominal field strength of the MRI scanners
      (``"3T"``, ``"7T"``... or ``"unknown"``): keeps the datasets with data at that strength
    - ``"age"`` : [min, max] | None ages of the participants (in years):
      keeps the datasets with at least one participant in this range (inclusive)
      (ages of the participants of neurobagel_participants.tsv)
    - ``"age_unknown"`` : bool keep the datasets without participant ages
      when filtering by age (default: True)
    - ``"sessions"`` : [min, max] | None number of sessions (inclusive)
    - ``"runs"`` : [min, max] | None number of runs of the first participant,
      across tasks and datatypes (inclusive)

    If ``None`` is passed will default to the DEFAULT_CONFIG.

    Returns
    -------
    pd.DataFrame
        Filtered listing of datasets.
    """
    ALL_TRUE = df["name"].apply(lambda x: bool(x))

    config = _check_config(config)

    mask_sources = ALL_TRUE
    sources = config["sources"]
    if sources is not None:
        mask_sources = df["source"].apply(lambda x: len({x}.intersection(sources)) > 0)
        if config["sources_and_or"] == "AND":
            mask_sources = df["source"].apply(
                lambda x: len({x}.intersection(sources)) == len(sources)
            )

    # better filtering should make sure
    # that the one of the requested datatypes has the task of interest
    mask_text = ALL_TRUE
    # keys built from the names of the text filters
    text_config: dict[str, Any] = dict(config)
    for key, column in TEXT_FILTERS.items():
        if text_config[key]:
            mask_text = mask_text & _text_mask(
                df[column], text_config[key], text_config[f"{key}_regex"]
            )

    mask_physio = ALL_TRUE
    if config["physio"] is not None:
        mask_physio = df["has_physio"] == config["physio"]

    mask_fmriprep = ALL_TRUE
    if config["fmriprep"] is not None:
        mask_fmriprep = df["has_fmriprep"] == config["fmriprep"]

    mask_mriqc = ALL_TRUE
    if config["mriqc"] is not None:
        mask_mriqc = df["has_mriqc"] == config["mriqc"]

    mask_participants = ALL_TRUE
    if config["participants"] is not None:
        mask_participants = df["useful_participants_tsv"] == config["participants"]

    mask_species = ALL_TRUE
    species = config["species"]
    if species is not None:
        mask_species = df["species"].apply(lambda x: set(x or [NOT_SPECIFIED]) <= set(species))

    mask_age = _age_mask(df, config["age"], config["age_unknown"])
    mask_sessions = _range_mask(df, "nb_sessions", config["sessions"])
    # unknown number of runs: as 0
    mask_runs = _range_mask(df, "nb_runs", config["runs"])

    mask_field_strength = ALL_TRUE
    if field_strength := config["field_strength"]:
        mask_field_strength = df["field_strengths"].apply(lambda x: field_strength in x)

    datatypes = config["datatypes"]
    mask_datatypes = df["datatypes"].apply(lambda x: len(set(x).intersection(datatypes)) > 0)
    if config["datatypes_and_or"] == "AND":
        mask_datatypes = df["datatypes"].apply(
            lambda x: len(set(x).intersection(datatypes)) == len(datatypes)
        )

    all_filters = pd.concat(
        (
            mask_text,
            mask_physio,
            mask_fmriprep,
            mask_mriqc,
            mask_participants,
            mask_datatypes,
            mask_sources,
            mask_species,
            mask_field_strength,
            mask_age,
            mask_sessions,
            mask_runs,
        ),
        axis=1,
    ).all(axis=1)

    filtered_df = df[all_filters]

    return filtered_df


def _age_mask(df: pd.DataFrame, age: list[float] | None, age_unknown: bool) -> pd.Series:
    """Keep the datasets with at least one participant in the age range (inclusive).

    And those without participant ages if ``age_unknown``.
    """
    if age is None:
        return pd.Series(True, index=df.index)
    min_age, max_age = age

    def in_range(ages: Any) -> bool:
        if not isinstance(ages, list) or not ages:
            return bool(age_unknown)
        return any(min_age <= x <= max_age for x in ages)

    return df["ages"].apply(in_range)


def _range_mask(df: pd.DataFrame, column: str, value_range: list[int] | None) -> pd.Series:
    """Keep the datasets whose value (number of sessions...) is in the range (inclusive).

    Missing values are counted as 0.
    """
    if value_range is None:
        return pd.Series(True, index=df.index)
    return df[column].fillna(0).between(value_range[0], value_range[1])


def _check_config(config: None | dict[str, Any]) -> FilterConfig:
    if not config:
        config = {}

    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary or None.")

    # work on a copy to not modify the input or DEFAULT_CONFIG
    config = {**DEFAULT_CONFIG, **config}

    for key, value in config.items():
        # "true" is a valid text to search
        if key not in TEXT_FILTERS:
            config[key] = booleanify(value)

    if isinstance(config["datatypes"], bool):
        config["datatypes"] = KNOWN_DATATYPES
    if isinstance(config["datatypes"], str):
        config["datatypes"] = [config["datatypes"]]

    if isinstance(config["sources"], str):
        config["sources"] = [config["sources"]]
    if isinstance(config["species"], str):
        config["species"] = [config["species"]]

    return cast(FilterConfig, config)


def booleanify(value: bool | str | list[str] | None) -> bool | str | list[str] | None:
    if value == "both":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    return value


def save_dataset_listing(df: pd.DataFrame) -> None:
    df["DatasetID"] = df["name"].copy()
    df = df.rename(
        columns={
            "nb_subjects": "NumMatchingSubjects",
            "datatypes": "AvailableImageModalites",
            "raw": "PortalURI",
            "name": "DatasetName",
        }
    )
    output_df = df[
        [
            "DatasetID",
            "PortalURI",
            "NumMatchingSubjects",
            "AvailableImageModalites",
            "sessions",
            "tasks",
            "fmriprep",
            "mriqc",
        ]
    ].copy()

    output_file = Path.cwd() / "datasets_results.tsv"
    output_df.to_csv(output_file, sep="\t", index=False)
    cc_log.info(f"Dataset listing saved to: {output_file}")


def _count_extensions(df: pd.DataFrame, datatype: str) -> pd.DataFrame:
    if datatype == "eeg":
        file_formats: dict[str, float] = {"bdf": 0, "edf": 0, "eeg": 0, "set": 0}
    elif datatype == "ieeg":
        file_formats = {"nwb": 0, "edf": 0, "eeg": 0, "set": 0, "mefd": 0}
    elif datatype == "meg":
        file_formats = {".ds": 0, "": 0, ".fif": 0, ".con": 0, ".kdf": 0, ".raw.mhd": 0}
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    datatype_df = df[df[datatype]]

    for row in datatype_df.iterrows():
        for key, value in row[1][f"{datatype}_file_formats"].items():
            if value > 0:
                file_formats[key] += 1

    for key, value in file_formats.items():
        file_formats[key] = value / len(datatype_df) * 100

    data: dict[str, list[str | float]] = {"extension": [], "count": []}
    for key, value in file_formats.items():
        data["extension"].append(key)
        data["count"].append(value)

    return pd.DataFrame(data)
