"""BIDS datasets dashboard, as a marimo notebook.

Opened by ``bids_inventory browse`` (``bids_inventory browse --edit`` to edit it).

Needs: pip install 'bids-inventory[browse]'

Also deployed on GitHub Pages, where it runs in the browser (WebAssembly):
see .github/workflows/deploy_dashboard.yml.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="BIDS inventory")


@app.cell
async def _():
    import sys

    if sys.platform == "emscripten":
        # in the browser (WebAssembly, on GitHub Pages):
        # bids_inventory is not on PyPI, install the wheel deployed next to the notebook
        import marimo as _mo
        import micropip
        from pyodide.http import pyfetch

        _public = _mo.notebook_location() / "public"
        _wheel = (await (await pyfetch(str(_public / "wheel.txt"))).string()).strip()
        await micropip.install([str(_public / _wheel), "plotly"])
    bids_inventory_installed = True
    return (bids_inventory_installed,)


@app.cell
def _(bids_inventory_installed):
    assert bids_inventory_installed
    import contextlib
    import tempfile
    from pathlib import Path

    import marimo as mo

    from bids_inventory._plotting import (
        datatypes_histogram,
        histogram_tasks,
        institutions_map,
        neurobagel_age_distribution,
        neurobagel_bar_per_dataset,
        neurobagel_female_ratio_vs_participants,
        neurobagel_per_dataset,
        neurobagel_unique_participants,
        plot_dataset_size_vs_time,
        scatter_subject_vs,
    )
    from bids_inventory._version import version
    from bids_inventory.institutions import (
        count_institutions,
        institutions_locations,
        load_institutions,
    )
    from bids_inventory.listing import (
        KNOWN_DATATYPES,
        known_datasets_df,
        neurobagel_participants_df,
    )
    from bids_inventory.wrangling import (
        NOT_SPECIFIED,
        filter_data,
        save_dataset_listing,
        wrangle_data,
    )

    return (
        KNOWN_DATATYPES,
        NOT_SPECIFIED,
        Path,
        contextlib,
        count_institutions,
        datatypes_histogram,
        filter_data,
        histogram_tasks,
        institutions_locations,
        institutions_map,
        known_datasets_df,
        load_institutions,
        mo,
        neurobagel_age_distribution,
        neurobagel_bar_per_dataset,
        neurobagel_female_ratio_vs_participants,
        neurobagel_participants_df,
        neurobagel_per_dataset,
        neurobagel_unique_participants,
        plot_dataset_size_vs_time,
        save_dataset_listing,
        scatter_subject_vs,
        tempfile,
        version,
        wrangle_data,
    )


@app.cell
def _(
    NOT_SPECIFIED,
    known_datasets_df,
    load_institutions,
    neurobagel_participants_df,
    wrangle_data,
):
    df = wrangle_data(known_datasets_df())
    SOURCES = sorted(df["source"].unique().tolist())
    # a dataset may have several species
    SPECIES = sorted({x for species in df["species"] for x in species} | {NOT_SPECIFIED})
    institutions = load_institutions()
    neurobagel_participants = neurobagel_participants_df()
    return SOURCES, SPECIES, df, institutions, neurobagel_participants


@app.cell
def _(mo):
    mo.md("""
    <h1 style="text-align: center">BIDS inventory: BIDS datasets dashboard</h1>
    """)
    return


@app.cell
def _(KNOWN_DATATYPES, NOT_SPECIFIED, SOURCES, SPECIES, mo):
    def checklist(options: list[str]) -> mo.ui.array:
        """Return a list of checkboxes, all checked."""
        return mo.ui.array([mo.ui.checkbox(label=x, value=True) for x in options])

    def true_false_both() -> mo.ui.radio:
        return mo.ui.radio(options=["true", "false", "both"], value="both", inline=True)

    datatypes = checklist(KNOWN_DATATYPES)
    datatypes_and_or = mo.ui.radio(options=["AND", "OR"], value="OR", inline=True)
    task = mo.ui.text(value="")
    sources = checklist(SOURCES)
    sources_and_or = mo.ui.radio(options=["AND", "OR"], value="OR", inline=True)
    fmriprep = true_false_both()
    mriqc = true_false_both()
    physio = true_false_both()
    participants = true_false_both()
    # most datasets whose species is not specified are about humans
    species = mo.ui.array(
        [mo.ui.checkbox(label=x, value=x in {"homo sapiens", NOT_SPECIFIED}) for x in SPECIES]
    )
    return (
        datatypes,
        datatypes_and_or,
        fmriprep,
        mriqc,
        participants,
        physio,
        sources,
        sources_and_or,
        species,
        task,
    )


@app.cell
def _(
    datatypes,
    datatypes_and_or,
    fmriprep,
    mo,
    mriqc,
    participants,
    physio,
    sources,
    sources_and_or,
    species,
    task,
):
    def section(title: str, description: str | None, *elements: object) -> mo.Html:
        return mo.vstack(
            [mo.md(f"## {title}"), *([mo.md(description)] if description else []), *elements]
        )

    mo.vstack(
        [
            section(
                "datatypes",
                "Keep datasets for specific datatypes",
                mo.hstack(
                    [mo.vstack(list(datatypes)), datatypes_and_or],
                    justify="start",
                    align="start",
                ),
            ),
            section("task", None, task),
            section(
                "sources",
                "Keep datasets from...",
                mo.hstack(
                    [mo.vstack(list(sources)), sources_and_or], justify="start", align="start"
                ),
            ),
            section(
                "species",
                """Keep datasets about these species: a dataset is kept only if all its species
                are selected (for example a dataset with humans and a phantom needs both).
                The species are from the species column of the participants.tsv,
                or from bids_inventory/data/species.tsv for some datasets without one.""",
                mo.vstack(list(species)),
            ),
            mo.hstack(
                [
                    section("fmriprep", "Keep datasets with fmriprep data", fmriprep),
                    section("mriqc", "Keep datasets with mriqc data", mriqc),
                    section("physio", "Keep datasets with physiological data", physio),
                    section(
                        "participants", "Keep only datasets with participants.tsv.", participants
                    ),
                ],
                justify="space-between",
            ),
        ]
    )
    return


@app.cell
def _(
    KNOWN_DATATYPES,
    SOURCES,
    SPECIES,
    datatypes,
    datatypes_and_or,
    df,
    filter_data,
    fmriprep,
    mriqc,
    participants,
    physio,
    sources,
    sources_and_or,
    species,
    task,
):
    filtered_df = filter_data(
        df,
        config={
            "datatypes": [x for x, keep in zip(KNOWN_DATATYPES, datatypes.value) if keep],
            "datatypes_and_or": datatypes_and_or.value,
            "sources": [x for x, keep in zip(SOURCES, sources.value) if keep],
            "sources_and_or": sources_and_or.value,
            "task": task.value,
            "fmriprep": fmriprep.value,
            "mriqc": mriqc.value,
            "physio": physio.value,
            "participants": participants.value,
            "species": [x for x, keep in zip(SPECIES, species.value) if keep],
        },
    )
    return (filtered_df,)


@app.cell
def _(Path, contextlib, filtered_df, mo, save_dataset_listing, tempfile):
    def table_to_show(dataframe):
        sub_df = dataframe[["name", "nb_subjects", "nb_sessions", "datatypes", "tasks"]].copy()
        # convert list to string
        sub_df["datatypes"] = sub_df["datatypes"].apply(lambda x: ", ".join(x))
        sub_df["tasks"] = sub_df["tasks"].apply(lambda x: ", ".join(x))
        return sub_df

    def dataset_listing() -> bytes:
        """Return the datasets_results.tsv that 'bids_inventory browse' saves."""
        with tempfile.TemporaryDirectory() as tmp_dir, contextlib.chdir(tmp_dir):
            save_dataset_listing(filtered_df.copy())
            return (Path(tmp_dir) / "datasets_results.tsv").read_bytes()

    mo.vstack(
        [
            mo.md("---"),
            mo.ui.table(
                table_to_show(filtered_df),
                page_size=15,
                selection=None,
                show_column_summaries=False,
            ),
            mo.download(
                data=dataset_listing,
                filename="datasets_results.tsv",
                label=f"Download the listing of the {len(filtered_df)} datasets",
            ),
        ]
    )
    return


@app.cell
def _(datatypes_histogram, filtered_df, mo):
    mo.vstack([mo.md("---"), datatypes_histogram(filtered_df)])
    return


@app.cell
def _(mo):
    subject_vs = mo.ui.radio(
        options=["number of tasks", "mean size per subject", "mean duration per subject"],
        value="number of tasks",
        inline=True,
    )
    return (subject_vs,)


@app.cell
def _(filtered_df, mo, scatter_subject_vs, subject_vs):
    y_columns = {
        "number of tasks": "nb_tasks",
        "mean size per subject": "mean_size",
        "mean duration per subject": "total_duration",
    }
    mo.vstack(
        [
            mo.md("---"),
            scatter_subject_vs(
                filtered_df,
                y=y_columns[subject_vs.value],
                size=None,
                color="source",
                title=f"{subject_vs.value} VS number of participants",
            ),
            mo.hstack([subject_vs], justify="center"),
        ]
    )
    return


@app.cell
def _(filtered_df, histogram_tasks, mo):
    mo.vstack([mo.md("---"), histogram_tasks(filtered_df)])
    return


@app.cell
def _(filtered_df, mo, plot_dataset_size_vs_time):
    mo.vstack([mo.md("---"), plot_dataset_size_vs_time(filtered_df)])
    return


@app.cell
def _(
    count_institutions,
    filtered_df,
    institutions,
    institutions_locations,
    institutions_map,
    mo,
):
    mo.vstack(
        [
            mo.md("---"),
            institutions_map(institutions_locations(institutions, count_institutions(filtered_df))),
        ]
    )
    return


@app.cell
def _(filtered_df, neurobagel_participants, neurobagel_unique_participants):
    # participants of the selected datasets annotated by Neurobagel
    nb_participants = neurobagel_unique_participants(
        neurobagel_participants[neurobagel_participants["dataset_id"].isin(filtered_df["name"])]
    )
    return (nb_participants,)


@app.cell
def _(mo, nb_participants):
    def _count(mask) -> str:
        return f"{mask.sum()} ({mask.mean() * 100:0.1f} %)" if len(mask) else "0"

    mo.md(f"""
    ---

    ## Participants of the datasets annotated by Neurobagel

    Participants of the selected datasets whose participants.tsv was annotated by
    [Neurobagel](https://neurobagel.org/) (OpenNeuro datasets only),
    with their age and sex standardized across datasets.

    - number of datasets: {nb_participants["dataset_id"].nunique()}
    - number of participants: {len(nb_participants)}
    - females: {_count(nb_participants["sex"] == "female")}
    - males: {_count(nb_participants["sex"] == "male")}
    - missing age: {_count(nb_participants["age"].isna())}
    - missing sex: {_count(nb_participants["sex"].isna())}
    """)
    return


@app.cell
def _(mo, nb_participants, neurobagel_age_distribution):
    mo.vstack([neurobagel_age_distribution(nb_participants)])
    return


@app.cell
def _(mo):
    neurobagel_per_dataset_column = mo.ui.radio(
        options={
            "proportion of females": "female_ratio",
            "missing age": "missing_age",
            "missing sex": "missing_sex",
        },
        value="proportion of females",
        inline=True,
    )
    return (neurobagel_per_dataset_column,)


@app.cell
def _(
    mo,
    nb_participants,
    neurobagel_bar_per_dataset,
    neurobagel_female_ratio_vs_participants,
    neurobagel_per_dataset,
    neurobagel_per_dataset_column,
):
    _per_dataset = neurobagel_per_dataset(nb_participants)
    mo.vstack(
        [
            neurobagel_female_ratio_vs_participants(_per_dataset),
            neurobagel_bar_per_dataset(_per_dataset, neurobagel_per_dataset_column.value),
            mo.hstack([neurobagel_per_dataset_column], justify="center"),
        ]
    )
    return


@app.cell
def _(mo, version):
    mo.md(f"""
    bids_inventory version: {version}

    [Github repo](https://github.com/neurodatascience/bids_inventory)

    Dashboard maintained by the [origami lab](https://neurodatascience.github.io/)

    [Report a BUG](https://github.com/neurodatascience/bids_inventory/issues/new)
    """)
    return


if __name__ == "__main__":
    app.run()
