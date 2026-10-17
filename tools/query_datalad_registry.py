"""Query the DataLad registry for BIDS datasets that are not on OpenNeuro.

https://blog.datalad.org/posts/registry/
API: https://registry.datalad.org/openapi/

Saves, in the current directory:

- datalad_registry_bids.json: all the dataset URLs found, with their metadata;
- datalad_registry_bids.tsv: one row per dataset URL, with a summary of its BIDS metadata.

Usage:
    python tools/query_datalad_registry.py [--search QUERY]

Needs: pip install requests pandas
"""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

API = "https://registry.datalad.org/api/v2/dataset-urls"
DEFAULT_SEARCH = 'metadata[bids_dataset]:"" AND NOT url:openneuro'
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


def summary(dataset_url: dict[str, Any]) -> dict[str, Any]:
    """Summarize a dataset URL and its BIDS metadata."""
    bids = next(
        (
            m.get("extracted_metadata") or {}
            for m in dataset_url.get("metadata") or []
            if m.get("extractor_name") == "bids_dataset"
        ),
        {},
    )
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
    args = parser.parse_args()

    dataset_urls = query(args.search)
    Path("datalad_registry_bids.json").write_text(json.dumps(dataset_urls, indent=1))
    df = pd.DataFrame([summary(x) for x in dataset_urls]).sort_values("url")
    df.to_csv("datalad_registry_bids.tsv", sep="\t", index=False)
    print(
        f"{len(df)} dataset URLs ({df['ds_id'].nunique()} datasets) "
        "saved to datalad_registry_bids.json and datalad_registry_bids.tsv"
    )


if __name__ == "__main__":
    main()
