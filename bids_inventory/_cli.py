"""Command line interface for bids_inventory."""

import logging
import os
import sys
from argparse import ArgumentParser, HelpFormatter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rich_argparse import RichHelpFormatter

from bids_inventory._version import __version__
from bids_inventory.logger import add_log_files, bi_logger

bi_log = bi_logger()


def global_parser(formatter_class: type[HelpFormatter] = HelpFormatter) -> ArgumentParser:
    parser = ArgumentParser(
        prog="bids_inventory",
        description="Keep an inventory of publicly available BIDS datasets.",
        formatter_class=formatter_class,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        help="Choose a subcommand",
        required=True,
    )

    browse_parser = subparsers.add_parser(
        "browse",
        help="""Open a dashboard (a marimo app) in the browser
to browse, visualize and filter the listing of known datasets.
The filtered list of datasets can be downloaded as datasets_results.tsv.""",
        formatter_class=parser.formatter_class,
    )
    browse_parser.add_argument(
        "--edit",
        action="store_true",
        help="Open the dashboard notebook in marimo's edit mode.",
    )
    _add_verbosity(browse_parser)

    _add_institutions_parser(subparsers, parser.formatter_class)

    neurobagel_parser = subparsers.add_parser(
        "neurobagel",
        help="""Combine the participants.tsv of the datasets annotated by Neurobagel
(https://github.com/OpenNeuroDatasets-JSONLD) with standardized age and sex,
in bids_inventory/data/neurobagel_participants.tsv.
Only processes the new repositories and those not annotated yet, unless --reindex is used.""",
        formatter_class=parser.formatter_class,
    )
    neurobagel_parser.add_argument(
        "--reindex",
        action="store_true",
        help="Process all the repositories again, not only the new and not yet annotated ones.",
    )
    neurobagel_parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only process these datasets (for example: ds000001 ds000002).",
    )
    neurobagel_parser.add_argument(
        "--jobs", default=8, type=int, help="Number of parallel downloads (default: %(default)s)."
    )
    neurobagel_parser.add_argument(
        "--config",
        help="""
        YAML file with local settings: the GitHub token is read from 'auth.token_file'
        (unless the GITHUB_TOKEN environment variable is set).
        See bids_inventory/data/config.yml for the available settings.
        Can also be set with the BIDS_INVENTORY_CONFIG environment variable.
        """,
        required=False,
        type=Path,
    )
    _add_verbosity(neurobagel_parser)

    remove_parser = subparsers.add_parser(
        "remove",
        help="""Remove datasets from the listing,
for example so that the next 'bids_inventory update' indexes them again.""",
        formatter_class=parser.formatter_class,
    )
    remove_parser.add_argument(
        "datasets",
        nargs="+",
        help="Names of the datasets to remove, for example: ds000001 ds000002.",
    )
    remove_parser.add_argument(
        "--source",
        help="Which listing to remove them from.",
        choices=["openneuro", "non-openneuro"],
        default="openneuro",
    )
    _add_verbosity(remove_parser)

    update_parser = subparsers.add_parser(
        "update",
        help="Update listing of known BIDS datasets.",
        formatter_class=parser.formatter_class,
    )
    update_parser.add_argument(
        "--debug",
        action="store_true",
        help="Only runs the update for a few subset of datasets.",
    )
    update_parser.add_argument(
        "--jobs",
        help="Number of new datasets to install, and of NIfTI headers to read, in parallel.",
        required=False,
        default=4,
        type=int,
    )
    update_parser.add_argument(
        "--source",
        help="""
        Which listing to update: datasets on OpenNeuro,
        or the other datasets of bids_inventory/data/non_openneuro_datasets.yml
        (ABIDE, CoRR, CNeuroMod, DANDI, NEMAR...): the missing ones are first
        installed in the folder set by 'local_paths.non_openneuro' in the configuration.
        """,
        choices=["openneuro", "non-openneuro"],
        default="openneuro",
    )
    update_parser.add_argument(
        "--reindex",
        help="""
        Index again all the datasets, not only those missing from the listing,
        for example to fill new fields.
        Can be resumed: if interrupted, run the same command again.
        """,
        action="store_true",
    )
    update_parser.add_argument(
        "--durations",
        help="""
        Only get again the duration of the runs of these datatypes
        (and the number of channels of the eeg, ieeg, meg and emg runs),
        for the OpenNeuro datasets already listed and installed locally.
        """,
        nargs="+",
        choices=["func", "pet", "ieeg", "eeg", "meg", "nirs", "motion", "emg"],
    )
    update_parser.add_argument(
        "--fields",
        help="""
        Only get again these fields,
        for the datasets already listed and installed locally (of the --source listing).
        """,
        nargs="+",
        choices=[
            "readme_size",
            "bidsignore_size",
            "references_and_links",
            "hed_version",
            "keywords",
            "cog_atlas_id",
            "cogpo_id",
            "physio_recordings",
        ],
    )
    update_parser.add_argument(
        "--log-dir",
        help="""
        Folder where to save the warnings and the errors of the update,
        in 2 separate files.
        """,
        required=False,
        default=Path("logs"),
        type=Path,
    )
    update_parser.add_argument(
        "--config",
        help="""
        YAML file with local settings (paths to datasets, GitHub token file...).
        See bids_inventory/data/config.yml for the available settings.
        Can also be set with the BIDS_INVENTORY_CONFIG environment variable.
        """,
        required=False,
        type=Path,
    )
    _add_verbosity(update_parser)

    return parser


def _add_institutions_parser(subparsers: Any, formatter_class: Any) -> None:
    institutions_parser = subparsers.add_parser(
        "institutions",
        help="""Locate the institutions of the inventory (bids_inventory/data/institutions.json)
and regroup the different names of the same institution.""",
        formatter_class=formatter_class,
    )
    institutions_subparsers = institutions_parser.add_subparsers(
        dest="institutions_command",
        help="Choose a subcommand",
        required=True,
    )

    update_parser = institutions_subparsers.add_parser(
        "update",
        help="""Geocode the institution names of the inventory that are not in institutions.json yet
(with OpenStreetMap, 1 request per second).
The file is saved after each geocoded address: if stopped, run it again to continue.""",
        formatter_class=formatter_class,
    )
    update_parser.add_argument(
        "--debug",
        action="store_true",
        help="Only geocode a few addresses.",
    )
    _add_verbosity(update_parser)

    relocate_parser = institutions_subparsers.add_parser(
        "relocate",
        help="""Geocode again institutions found in the wrong place, only in their country
(from the country code at the end of DICOM addresses: "...,Magdeburg,District,DE,39120").
Without names: all the institutions with a country code that are located in another country.""",
        formatter_class=formatter_class,
    )
    relocate_parser.add_argument(
        "names",
        nargs="*",
        help="Institutions to geocode again (keys of institutions.json).",
    )
    relocate_parser.add_argument(
        "--country",
        help="Country code to search the institutions in (for example: ch), for names without one.",
    )
    _add_verbosity(relocate_parser)

    suggest_parser = institutions_subparsers.add_parser(
        "suggest",
        help="""Write the pairs of institutions that are probably the same to a TSV file.
Review it ('y' in the 'accept' column for the pairs to merge),
then merge them with 'bids_inventory institutions apply-matches'.""",
        formatter_class=formatter_class,
    )
    suggest_parser.add_argument(
        "--output",
        help="File where to save the pairs (default: %(default)s).",
        default=Path("institution_matches.tsv"),
        type=Path,
    )
    _add_verbosity(suggest_parser)

    apply_parser = institutions_subparsers.add_parser(
        "apply-matches",
        help="Merge the pairs of institutions accepted in the file written by 'suggest'.",
        formatter_class=formatter_class,
    )
    apply_parser.add_argument(
        "--matches",
        help="File with the reviewed pairs (default: %(default)s).",
        default=Path("institution_matches.tsv"),
        type=Path,
    )
    _add_verbosity(apply_parser)


def _run_institutions(args: Any) -> None:
    """Run the 'institutions' subcommands."""
    import pandas as pd

    from bids_inventory import load_inventory
    from bids_inventory.institutions import (
        INSTITUTIONS_FILE,
        clean_address,
        count_institutions,
        explode_unlocated,
        gather_unlocated,
        institutions_locations,
        load_institutions,
        save_institutions,
    )

    institutions = gather_unlocated(load_institutions())

    if args.institutions_command == "update":
        with _check_extra("institutions"):
            from bids_inventory._institutions_update import update_institutions

        names = count_institutions(load_inventory())["name"]
        update_institutions(institutions, names, debug=args.debug)
        bi_log.info(f"Saved {INSTITUTIONS_FILE}")
        return

    if args.institutions_command == "relocate":
        with _check_extra("institutions"):
            from bids_inventory._institutions_update import relocate_institutions

        if args.country and not args.names:
            bi_log.error("--country needs the names of the institutions to relocate")
            sys.exit(1)
        try:
            relocate_institutions(
                institutions,
                args.names or None,
                country=args.country.lower() if args.country else None,
            )
        except ValueError as e:
            bi_log.error(str(e))
            sys.exit(1)
        bi_log.info(f"Saved {INSTITUTIONS_FILE}")
        return

    with _check_extra("institutions"):
        from bids_inventory._institutions_matching import apply_matches, suggest_matches

    if args.institutions_command == "suggest":
        matches = suggest_matches(explode_unlocated(institutions), clean_address)
        matches.to_csv(args.output, index=False, sep="\t", lineterminator="\n")
        bi_log.info(
            f"Saved {len(matches)} possible matches to {args.output}: "
            "write 'y' in the 'accept' column of the ones to merge, "
            "then run 'bids_inventory institutions apply-matches'"
        )
        return

    if not args.matches.is_file():
        bi_log.error(f"File not found: {args.matches}. Run 'bids_inventory institutions suggest'.")
        sys.exit(1)
    locations = institutions_locations(institutions, count_institutions(load_inventory()))
    counts = dict(zip(locations["name"], locations["count"]))
    matches = pd.read_csv(args.matches, sep="\t", dtype=str, keep_default_na=False)
    merged = apply_matches(explode_unlocated(institutions), matches, counts)
    save_institutions(gather_unlocated(merged))
    bi_log.info(f"Saved {INSTITUTIONS_FILE}")


def _run_update(args: Any) -> None:
    """Run the 'update' subcommand."""
    with _check_extra("update"):
        from bids_inventory._update import (
            CONFIG_ENV_VAR,
            update,
            update_durations,
            update_fields,
        )

    if args.jobs < 1:
        bi_log.error(f"--jobs must be at least 1, got {args.jobs}")
        sys.exit(1)

    if args.config is not None:
        if not args.config.is_file():
            bi_log.error(f"Config file not found: {args.config}")
            sys.exit(1)
        os.environ[CONFIG_ENV_VAR] = str(args.config.resolve())

    warnings_log, errors_log = add_log_files(args.log_dir)
    try:
        if args.durations:
            update_durations(args.durations, debug=args.debug, jobs=args.jobs)
        elif args.fields:
            update_fields(args.fields, debug=args.debug, source=args.source)
        else:
            update(debug=args.debug, jobs=args.jobs, reindex=args.reindex, source=args.source)
    finally:
        for log_file in [warnings_log, errors_log]:
            if log_file.exists():
                bi_log.info(f"See {log_file}")


def _run_neurobagel(args: Any) -> None:
    """Run the 'neurobagel' subcommand."""
    with _check_extra("update"):
        from bids_inventory._neurobagel import update_neurobagel
        from bids_inventory._update import CONFIG_ENV_VAR, get_auth

    if args.config is not None:
        if not args.config.is_file():
            bi_log.error(f"Config file not found: {args.config}")
            sys.exit(1)
        os.environ[CONFIG_ENV_VAR] = str(args.config.resolve())

    # same token as 'bids_inventory update': GITHUB_TOKEN, or auth.token_file in the config
    auth = get_auth()
    update_neurobagel(
        token=auth[1] if auth else None,
        reindex=args.reindex,
        datasets=args.datasets,
        jobs=args.jobs,
    )


def _add_verbosity(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--verbosity",
        help="""
        Verbosity level.
        """,
        required=False,
        choices=[0, 1, 2, 3],
        default=2,
        type=int,
    )


def set_verbosity(verbosity: int) -> None:
    levels = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}
    bi_log.setLevel(levels[verbosity])

    if verbosity < 3:
        logging.getLogger("datalad").setLevel(logging.WARNING)
        logging.getLogger("datalad.gitrepo").setLevel(logging.ERROR)


@contextmanager
def _check_extra(extra: str) -> Iterator[None]:
    """Exit with an explanation if the optional dependencies of a command are missing."""
    try:
        yield
    except ImportError as e:
        bi_log.error(
            f"The '{extra}' command needs optional dependencies ({e.name} is missing).\n"
            f"Install them with: pip install 'bids-inventory[{extra}]'"
        )
        sys.exit(1)


def cli(argv: Sequence[str] = sys.argv) -> None:
    """Entry point."""
    parser = global_parser(formatter_class=RichHelpFormatter)
    args = parser.parse_args(argv[1:])

    set_verbosity(args.verbosity)

    if args.command == "browse":
        with _check_extra("browse"):
            from bids_inventory._browse import browse

        browse(edit=args.edit)

    elif args.command == "institutions":
        _run_institutions(args)

    elif args.command == "neurobagel":
        _run_neurobagel(args)

    elif args.command == "remove":
        from bids_inventory.listing import (
            _non_openneuro_listing,
            _openneuro_listing,
            remove_datasets,
        )

        listing = _openneuro_listing() if args.source == "openneuro" else _non_openneuro_listing()
        removed = remove_datasets(args.datasets, listing)
        bi_log.info(f"Removed {len(removed)} datasets from {listing}: {' '.join(removed)}")
        if not_found := sorted(set(args.datasets) - set(removed)):
            bi_log.warning(f"Not in the listing: {' '.join(not_found)}")

    elif args.command == "update":
        _run_update(args)
