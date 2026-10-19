# Tools

Helpers that are not part of the `bids_inventory` command line.
To update the listing, use `bids_inventory update`,
and for the institutions `bids_inventory institutions`: see the package README.

- `query_datalad_registry.py`:
  lists the BIDS datasets of the DataLad registry that are not on OpenNeuro and not tracked yet
  (see `bids_inventory/data/non_openneuro_datasets.yml`).
- `build_dashboard.sh [output_dir]`:
  builds the dashboard as a static website that runs in the browser (default: `_site`),
  as deployed on GitHub Pages.
