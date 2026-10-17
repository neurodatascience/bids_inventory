# bids-inventory

Keeps an inventory of publicly available BIDS datasets
(OpenNeuro and a few other sources)
with information about their content (datatypes, tasks, sessions, number of participants, scan duration...).

The inventory is stored in `bids_inventory/data/` as [JSON lines](https://jsonlines.org/) files
(one dataset per line):

- `openneuro.jsonl`
- `non_openneuro.jsonl`

A description of each field is in `bids_inventory/data/columns_description.json`.
It is generated from the data model in `bids_inventory/models.py`:
after changing the model, regenerate it with `python -m bids_inventory.models`.

## Installation

```bash
pip install .
```

The base install is enough to load the listing.
The commands need optional dependencies:

- `bids-inventory[browse]`: `bids_inventory browse`, a dashboard to browse and filter the listing
- `bids-inventory[update]`: `bids_inventory update`, to update the listing,
  and `bids_inventory neurobagel`, to combine the participants of the datasets annotated by Neurobagel
- `bids-inventory[institutions]`: `bids_inventory institutions`, to locate the institutions

## Usage

```python
from bids_inventory import filter_data, load_inventory

# listing with extra summary columns (has_fmriprep, nb_tasks, total_duration...)
df = load_inventory()
eeg_datasets = filter_data(df, config={"datatypes": ["eeg"]})
# text filters (case insensitive) on the tasks, authors, institutions and their countries,
# with regular expressions if the "..._regex" key is True
canadian_rest = filter_data(
    df, config={"task": "^rest$", "task_regex": True, "country": "canada"}
)
```

The institutions are named as in the keys of `bids_inventory/data/institutions.json`
(`institution_names` column), and their countries are in the `countries` column.

`field_strengths` has the nominal field strengths of the MRI scanners (`3T`, `7T`...),
`unknown` for MRI data without `MagneticFieldStrength` in the sidecars:
`filter_data(df, config={"field_strength": "7T"})`.

To browse the listing in a dashboard (a [marimo](https://marimo.io/) app) in the browser:

```bash
pip install '.[browse]'
bids_inventory browse
```

The filtered list of datasets can be downloaded from it as `datasets_results.tsv`.

The dashboard is also deployed on GitHub Pages, where it runs in the browser (WebAssembly):
https://neurodatascience.github.io/bids_inventory/
(by the `deploy_dashboard.yml` workflow, after each change on `main`).
To build it locally: `tools/build_dashboard.sh` (needs `pip install build marimo uv`).
It also shows the institutions of the filtered datasets on a world map
(from `bids_inventory/data/institutions.json`, see [Institutions](#institutions)).
The dashboard is the marimo notebook `bids_inventory/browse_app.py`:
`bids_inventory browse --edit` opens it in edit mode.

## Updating the inventory

```bash
pip install -e '.[update]'
bids_inventory update
```

The OpenNeuro listing is updated every week by the `update_listing.yml` GitHub workflow.

Everything is done with `bids_inventory update` (see `bids_inventory update --help`):

| command | what it does |
|---|---|
| `bids_inventory update` | installs and indexes the OpenNeuro datasets that are not in the listing yet |
| `bids_inventory update --reindex` | indexes again all the OpenNeuro datasets, for example after new fields were added to the listing (also installs the datasets missing locally) |
| `bids_inventory update --durations eeg meg` | only gets again the duration of the runs of these datatypes, for the OpenNeuro datasets already listed and installed locally |
| `bids_inventory update --source non-openneuro` | indexes the datasets that are not on OpenNeuro (ABIDE, ABIDE 2, ADHD-200, CoRR, CNeuroMod) that are not in `non_openneuro.jsonl` yet |
| `bids_inventory update --source non-openneuro --reindex` | indexes again all of them |
| `bids_inventory remove ds000001 ds000002` | removes datasets from the listing, so that the next `bids_inventory update` indexes them again (`--source non-openneuro` for the other listing) |

Useful options:

- `--config FILE`: your local settings (see below).
- `--jobs N`: number of datasets to install, and of NIfTI headers to read, in parallel (default: 4).
- `--log-dir DIR`: where warnings and errors are saved, in 2 separate files (default: `logs`).
  Each warning and error mentions the dataset it comes from, for example `[ds004000 171/1380]`.

The listing is saved after each dataset is indexed:
if an update is stopped, run the same command again to continue where it stopped.
With `--reindex`, each entry is replaced in place, so the listing stays complete.

The durations of the runs are those of the first participant:
from the NIfTI headers for func,
and from the JSON sidecars for the other datatypes
(`RecordingDuration` for eeg, ieeg, meg, nirs and motion, the sum of `FrameDuration` for pet),
or else from the headers of the files when they can be read (NIfTI for pet, BrainVision for eeg and ieeg).

### Configuration

The update needs a few local settings.
Do not edit `bids_inventory/data/config.yml`:
it only contains the default (empty) settings and documents each of them.
Instead create your own YAML file, outside of the repository,
with only the settings you need:

```yaml
# ~/bids_inventory_config.yml
auth:
    # text file containing a GitHub token (or set the GITHUB_TOKEN environment variable)
    token_file: ~/tokens/github_token.txt
local_paths:
    datalad:
        # openneuro folder of the datalad superdataset
        OpenNeuroDatasets: ~/datalad/datasets.datalad.org/openneuro
        # folder of the datalad superdataset of the OpenNeuro derivatives
        OpenNeuroDerivatives: ~/datalad/datasets.datalad.org/openneuro-derivatives
    openneuro:
        # where the OpenNeuro datasets missing from the superdataset are installed
        OpenNeuroDatasets: ~/openneuro
    # where the datasets that are not on OpenNeuro are installed
    # (only needed for --source non-openneuro)
    non_openneuro: ~/non_openneuro
```

and pass it with `--config ~/bids_inventory_config.yml`,
or set the `BIDS_INVENTORY_CONFIG` environment variable to its path.
If a setting needed by the update is missing, the error message says which one.

### Installing the datasets

The datasets are installed with [datalad](https://www.datalad.org/):
only their git history and file names are downloaded, not the data.

**OpenNeuro**: the update reads the datasets from the datalad superdataset,
which must be installed (the update stops otherwise):

```bash
cd ~/datalad
datalad install ///
cd datasets.datalad.org
datalad install openneuro openneuro-derivatives
```

Then `bids_inventory update`:

1. installs, in the superdataset, the datasets it needs that are not installed there yet
   (you can also install all of them beforehand:
   `cd openneuro && datalad -f '{path}' subdatasets | xargs -n 1 -P 10 datalad install`),
2. installs the datasets that are missing from the superdataset
   (for example datasets more recent than the superdataset)
   in `local_paths.openneuro.OpenNeuroDatasets`,
3. indexes the datasets of the superdataset, then the others.

The derivatives (fMRIPrep, MRIQC) are listed from their superdataset first,
then from GitHub for those missing from it.
The version of the pipeline is read from the superdataset when the derivative dataset is installed there,
otherwise from GitHub.

**Other datasets**: install them with `tools/install_extra_datasets.sh`,
in the folder set by `local_paths.non_openneuro`:

```bash
tools/install_extra_datasets.sh ~/non_openneuro
```

There are many datasets: make sure you have enough disk space.

### Skipping datasets

Datasets listed in `bids_inventory/data/skipped_datasets.csv` are never installed or indexed,
for example datasets deleted from OpenNeuro that are still mirrored on GitHub
(their file contents cannot be downloaded).
Each dataset is listed with the reason why it is skipped.

### GitHub API tokens

Listing the datasets of the OpenNeuro GitHub organizations needs several calls to the GitHub API,
which limits the number of requests per hour.
To increase this limit, create a GitHub API token and either:

- set the `GITHUB_TOKEN` environment variable (it takes precedence),
- or save it to a text file (not in the repository)
  and set `auth.token_file` to its path in your configuration file.

## Institutions

`bids_inventory/data/institutions.json` gives the location of the institutions
found in the JSON sidecars of the datasets:

```json
"name": {
    "AlternativeNames": ["other names of the same institution found in datasets"],
    "Address": "",
    "City": "",
    "Country": "",
    "Latitude": null,
    "Longitude": null
}
```

The names that could not be located are the alternative names of the `unlocated` entry.
This file can be edited by hand, to fix a location or to group the names of an institution:
the commands below never change the institutions already in it,
they only add new names.

```bash
pip install -e '.[institutions]'

# geocode the institution names of the inventory that are not in the file yet
# (OpenStreetMap, 1 request per second; if stopped, run it again to continue)
bids_inventory institutions update

# regroup the names of the same institution:
# write the pairs that are probably the same to institution_matches.tsv,
bids_inventory institutions suggest
# review it ('y' in the 'accept' column of the pairs to merge), then merge them
bids_inventory institutions apply-matches
```

## Participants of the datasets annotated by Neurobagel

`bids_inventory/data/neurobagel_participants.tsv` combines the `participants.tsv`
of the datasets of [OpenNeuroDatasets-JSONLD](https://github.com/OpenNeuroDatasets-JSONLD)
annotated by [Neurobagel](https://neurobagel.org/),
with the age (in years) and the sex (`female`, `male`, `other`) standardized across datasets
using the annotations of their `participants.json`:

```python
from bids_inventory.listing import neurobagel_participants_df

participants = neurobagel_participants_df()
```

`bids_inventory/data/neurobagel_datasets.tsv` (`neurobagel_datasets_df()`) lists the repositories:
used or not, why, the problems found in their annotations (`notes` column),
and the species of their participants (`species` column).
Some datasets are not about humans (mice, rats, sheep...),
and the dashboard only shows those about humans by default.
The species comes from the `species` column of `participants.tsv`;
most datasets do not have one (`species` is then empty),
so the species of some of them is listed in `bids_inventory/data/species.tsv`:
add the non-human datasets you find there.

To update them:

```bash
pip install -e '.[update]'
# only processes the new repositories and those not annotated yet
# (--reindex to process all of them again)
bids_inventory neurobagel --config my_config.yml
```

The GitHub token is the same as for `bids_inventory update`
(see [GitHub API tokens](#github-api-tokens)):
the `GITHUB_TOKEN` environment variable, or `auth.token_file` in the configuration file.
The downloaded files are cached in `tmp/neurobagel_cache` (ignored by git).
