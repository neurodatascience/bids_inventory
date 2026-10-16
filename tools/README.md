# Tools

Helpers that are not part of the `bids_inventory` command line.
To update the listing, use `bids_inventory update`,
and for the institutions `bids_inventory institutions`: see the package README.

- `install_extra_datasets.sh [target_dir]`:
  installs the datasets that are not on OpenNeuro (ABIDE, ABIDE 2, ADHD-200, CoRR, CNeuroMod)
  in `target_dir` (default: `./tmp`).
  Then set `local_paths.non_openneuro` to `target_dir` in your configuration
  and run `bids_inventory update --source non-openneuro`.
- `build_dashboard.sh [output_dir]`:
  builds the dashboard as a static website that runs in the browser (default: `_site`),
  as deployed on GitHub Pages.
- `combine_neurobagel_participants.py`:
  combines the `participants.tsv` of the datasets of
  [OpenNeuroDatasets-JSONLD](https://github.com/OpenNeuroDatasets-JSONLD)
  annotated by Neurobagel into a single TSV,
  with the age (in years) and the sex (female, male, other) standardized across datasets
  using the annotations of their `participants.json`
  (`python tools/combine_neurobagel_participants.py --token-file <file with a GitHub token>`).
  It also writes a summary of the datasets: used or not, why, and problems found in the annotations.
  Both are saved in `bids_inventory/data/` (`neurobagel_participants.tsv`, `neurobagel_datasets.tsv`).
  Running it again only processes the new repositories and those not annotated yet
  (`--reindex` to process all of them again).
