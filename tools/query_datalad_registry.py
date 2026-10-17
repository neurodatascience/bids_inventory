"""Query the DataLad registry for BIDS datasets that are not on OpenNeuro.

https://blog.datalad.org/posts/registry/
API: https://registry.datalad.org/openapi/

Keeps the raw datasets (DatasetType "raw", or not specified: "raw" by default)
with at least one subject, that are not copies of OpenNeuro datasets: the ones with
an OpenNeuro id in their URL (for example the NEMAR mirrors: nemarDatasets/on005628),
an OpenNeuro DOI or link in their dataset_description.json,
or (with --openneuro-dir) the same name as an OpenNeuro dataset installed locally.

Saves, in the current directory:

- datalad_registry_bids.json: the dataset URLs kept, with their metadata;
- datalad_registry_bids.tsv: one row per dataset URL kept,
  with a summary of its BIDS metadata;
- datalad_registry_openneuro_copies.tsv: the same for all the copies of OpenNeuro datasets
  found (whatever their type and number of subjects), with the id of the OpenNeuro dataset.

Usage:
    python tools/query_datalad_registry.py [--search QUERY] [--openneuro-dir DIR]

Needs: pip install requests pandas
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

API = "https://registry.datalad.org/api/v2/dataset-urls"
# openfmri: the former OpenNeuro, courtois-neuromod: already in the inventory,
# bids_apps_test_data: test data
DEFAULT_SEARCH = (
    'metadata[bids_dataset]:"" AND NOT url:openneuro AND NOT url:openfmri '
    "AND NOT url:courtois-neuromod AND NOT url:bids_apps_test_data"
)
# all the copies of OpenNeuro datasets found
COPIES_FILE = "datalad_registry_openneuro_copies.tsv"
# fields of the dataset_description.json that can mention the OpenNeuro dataset
OPENNEURO_FIELDS = ["DatasetDOI", "ReferencesAndLinks", "HowToAcknowledge", "SourceDatasets"]
PER_PAGE = 100


def query(search: str) -> list[dict[str, Any]]:
    """Return all the dataset URLs matching a search, with their metadata."""
    session = requests.Session()
    params: dict[str, Any] = {
        "search": search,
        "per_page": PER_PAGE,
        "return_metadata": "content",
        "page": 1,
    }
    dataset_urls: list[dict[str, Any]] = []
    while True:
        response = session.get(API, params=params, timeout=120)
        response.raise_for_status()
        page = response.json()
        dataset_urls.extend(page["dataset_urls"])
        print(f"page {page['cur_pg_num']}: {len(dataset_urls)} dataset URLs")
        if not page.get("next_pg"):
            return dataset_urls
        params["page"] += 1


def bids_metadata(dataset_url: dict[str, Any]) -> dict[str, Any]:
    """Return the metadata of a dataset URL extracted by the bids_dataset extractor."""
    return next(
        (
            m.get("extracted_metadata") or {}
            for m in dataset_url.get("metadata") or []
            if m.get("extractor_name") == "bids_dataset"
        ),
        {},
    )


def openneuro_names(openneuro_dir: Path) -> dict[str, str]:
    """Map the (lower case) names of the OpenNeuro datasets installed locally to their id."""
    names: dict[str, str] = {}
    for description in sorted(openneuro_dir.glob("ds*/dataset_description.json")):
        try:
            name = json.loads(description.read_text()).get("Name")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(name, str) and len(name.strip()) > 3:
            names.setdefault(name.strip().lower(), description.parent.name)
    return names


def openneuro_id(dataset_url: dict[str, Any], names: dict[str, str]) -> str:
    """Return the id of the OpenNeuro dataset a dataset URL is a copy of (empty if none).

    From its URL (nemarDatasets/on005628), an OpenNeuro DOI or link
    in its dataset_description.json, or its name (``names``: see ``openneuro_names``).
    """
    if match := re.search(r"/(?:ds|on|nm)(\d{6})", dataset_url["url"]):
        return f"ds{match.group(1)}"
    bids = bids_metadata(dataset_url)
    text = json.dumps({field: bids.get(field) for field in OPENNEURO_FIELDS})
    if match := re.search(r"openneuro[./a-z]*?(ds\d{6})", text, re.IGNORECASE):
        return match.group(1)
    return names.get(str(bids.get("Name") or "").strip().lower(), "")


def is_raw(dataset_url: dict[str, Any]) -> bool:
    """Is it a raw dataset? DatasetType is "raw" by default."""
    return str(bids_metadata(dataset_url).get("DatasetType") or "raw").lower() == "raw"


def summary(dataset_url: dict[str, Any]) -> dict[str, Any]:
    """Summarize a dataset URL and its BIDS metadata."""
    bids = bids_metadata(dataset_url)
    entities = bids.get("entities") or {}
    url = urlparse(dataset_url["url"])
    return {
        "url": dataset_url["url"],
        "host": url.netloc,
        # for example the GitHub organization
        "owner": url.path.strip("/").split("/")[0] if url.path.strip("/") else "",
        "ds_id": dataset_url.get("ds_id"),
        "name": bids.get("Name"),
        "bids_version": bids.get("BIDSVersion"),
        # "raw" if not specified, but derivatives do not always specify it
        "dataset_type": bids.get("DatasetType"),
        "generated_by": ", ".join(
            str(x.get("Name")) for x in bids.get("GeneratedBy") or [] if isinstance(x, dict)
        ),
        "license": bids.get("License"),
        "nb_authors": len(bids.get("Authors") or []),
        "nb_subjects": len(entities.get("subject") or []),
        "nb_sessions": len(entities.get("session") or []),
        "datatypes": ", ".join(sorted(map(str, entities.get("datatype") or []))),
        "tasks": ", ".join(sorted(map(str, entities.get("task") or []))),
        "annexed_files_size": dataset_url.get("annexed_files_in_wt_size"),
        "annexed_files_count": dataset_url.get("annexed_files_in_wt_count"),
        "last_update": dataset_url.get("last_update_dt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--search", default=DEFAULT_SEARCH, help="Search query (web UI syntax).")
    parser.add_argument(
        "--openneuro-dir",
        type=Path,
        help="Folder with OpenNeuro datasets installed locally (ds*): "
        "datasets with the same name as one of them are excluded.",
    )
    args = parser.parse_args()

    names = openneuro_names(args.openneuro_dir) if args.openneuro_dir else {}
    dataset_urls = query(args.search)
    print(f"{len(dataset_urls)} dataset URLs found")

    # all the copies of OpenNeuro datasets found, whatever their type and subjects
    copies = pd.DataFrame(
        [
            {**summary(x), "openneuro_id": copy_of}
            for x in dataset_urls
            if (copy_of := openneuro_id(x, names))
        ]
    ).sort_values(["openneuro_id", "url"])
    copies.to_csv(COPIES_FILE, sep="\t", index=False)
    print(
        f"{len(copies)} copies of {copies['openneuro_id'].nunique()} OpenNeuro datasets "
        f"saved to {COPIES_FILE}"
    )

    for reason, keep in [
        # for example empty or toy datasets
        ("without subjects", lambda x: summary(x)["nb_subjects"] > 0),
        ("not raw (DatasetType)", is_raw),
        ("copies of OpenNeuro datasets", lambda x: not openneuro_id(x, names)),
    ]:
        kept = [x for x in dataset_urls if keep(x)]
        print(f"  {len(dataset_urls) - len(kept)} excluded: {reason}")
        dataset_urls = kept
    Path("datalad_registry_bids.json").write_text(json.dumps(dataset_urls, indent=1))
    df = pd.DataFrame([summary(x) for x in dataset_urls]).sort_values("url")
    df.to_csv("datalad_registry_bids.tsv", sep="\t", index=False)
    print(
        f"{len(df)} dataset URLs ({df['ds_id'].nunique()} datasets) "
        "saved to datalad_registry_bids.json and datalad_registry_bids.tsv"
    )


if __name__ == "__main__":
    main()
