import math

import pytest

# needs the 'update' extra
pytest.importorskip("datalad")
pytest.importorskip("mne")

from bids_inventory._update import (  # noqa: E402
    count_meg_file_formats,
    count_stim_files,
    created_on,
    get_authors,
    get_dataset_size,
    get_duration,
    get_info_dataset,
    get_institutions,
    get_license,
    get_meeg_header,
    get_nb_subjects,
    get_references_and_links,
    has_participant_tsv,
    list_data_files,
    list_datatypes,
    list_participants_tsv_columns,
    list_sessions,
    list_tasks,
)

# from bids_inventory._update import get_duration


def test_count_meg_file_formats(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert count_meg_file_formats(ds004276) == {
        ".ds": 0,
        "": 0,
        ".fif": 19,
        ".con": 0,
        ".kdf": 0,
        ".raw.mhd": 0,
    }


def test_count_stim_files(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert count_stim_files(ds004276) == 0


def test_created_on(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert created_on(ds004276) == "Fri Sep 23 14:41:43 2022 +0000"


def test_list_data_files(install_dataset):
    ds004276 = install_dataset("ds004276")
    sessions = list_sessions(ds004276)
    assert len(list_data_files(ds004276, sessions=sessions)) == 93


def test_list_participants_tsv_columns(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert list_participants_tsv_columns(ds004276 / "participants.tsv") == [
        "participant_id",
        "age",
        "sex",
        "hand",
    ]


def test_has_participant_tsv(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert has_participant_tsv(ds004276) == (True, True, ["participant_id", "age", "sex", "hand"])


def test_has_participant_tsv_empty(install_dataset):
    ds001339 = install_dataset("ds001339")
    assert has_participant_tsv(ds001339) == (False, False, [])


def test_list_sessions(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert list_sessions(ds004276) == ["20191007"]


def test_list_datatypes(install_dataset):
    ds004276 = install_dataset("ds004276")
    sessions = list_sessions(ds004276)
    assert list_datatypes(ds004276, sessions=sessions) == ["beh", "meg"]


def test_get_nb_subjects(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_nb_subjects(ds004276) == 19


def test_list_tasks(install_dataset):
    ds004276 = install_dataset("ds004276")
    sessions = list_sessions(ds004276)
    assert list_tasks(ds004276, sessions=sessions) == ["noise", "words"]


def test_get_license(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_license(ds004276) == "CC0"


def test_get_authors(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_authors(ds004276) == [
        "Phoebe Gaston",
        "Christian Brodbeck",
        "Colin Phillips",
        "Ellen Lau",
    ]


def test_get_institutions(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_institutions(ds004276) == []


def test_get_references_and_links_empty(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_references_and_links(ds004276) == []


def test_get_references_and_links_non_empty(install_dataset):
    ds001339 = install_dataset("ds001339")
    assert get_references_and_links(ds001339) == [
        (
            "The social brain is highly sensitive "
            "to the mere presence of social information: "
            "An automated meta-analysis and an independent study"
        )
    ]


def test_get_dataset_size(install_dataset):
    ds004276 = install_dataset("ds004276")
    assert get_dataset_size(ds004276) == "11.6 GB"


def test_get_info_dataset_smoke(install_dataset):
    ds004276 = install_dataset("ds004276")
    get_info_dataset(ds004276, "")


def test_get_duration(install_dataset):
    """ds001339 has uncompressed .nii files."""
    ds001339 = install_dataset("ds001339")
    sessions = list_sessions(ds001339)
    datatypes = list_datatypes(ds001339, sessions=sessions)
    tasks = list_tasks(ds001339, sessions=sessions)
    assert get_duration(ds001339, datatypes, tasks) == {"func": {"iaps": [(120, 2.0)] * 5}}


def test_get_duration_ds000002(install_dataset):
    ds000002 = install_dataset("ds000002")
    sessions = list_sessions(ds000002)
    datatypes = list_datatypes(ds000002, sessions=sessions)
    tasks = list_tasks(ds000002, sessions=sessions)
    durations = get_duration(ds000002, datatypes, tasks)
    assert list(durations["func"].keys()) == [
        "deterministicclassification",
        "mixedeventrelatedprobe",
        "probabilisticclassification",
    ]
    assert durations == {
        "func": {
            "deterministicclassification": [(180, 2.0), (180, 2.0)],
            "mixedeventrelatedprobe": [(237, 2.0), (237, 2.0)],
            "probabilisticclassification": [(180, 2.0), (180, 2.0)],
        }
    }


def test_get_duration_parallel(install_dataset):
    """Reading headers in parallel must give the same durations in the same order."""
    ds000002 = install_dataset("ds000002")
    sessions = list_sessions(ds000002)
    datatypes = list_datatypes(ds000002, sessions=sessions)
    tasks = list_tasks(ds000002, sessions=sessions)
    assert get_duration(ds000002, datatypes, tasks, jobs=4) == get_duration(
        ds000002, datatypes, tasks, jobs=1
    )


def test_get_duration_ds002041(install_dataset):
    ds002041 = install_dataset("ds002041")
    sessions = list_sessions(ds002041)
    datatypes = list_datatypes(ds002041, sessions=sessions)
    tasks = list_tasks(ds002041, sessions=sessions)
    get_duration(ds002041, datatypes, tasks)


def test_get_meeg_header_unreadable(tmp_path, caplog):
    """A malformed header should be skipped with a warning, not stop the update."""
    from datalad import api

    ds = api.create(tmp_path / "ds", result_renderer="disabled")
    eeg = ds.pathobj / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    header = eeg / "sub-01_task-rest_eeg.vhdr"
    header.write_bytes(b"\x00\x05\x16\x07\x00\x02Mac OS X")
    header.with_suffix(".vmrk").write_bytes(b"")
    ds.save(result_renderer="disabled")

    assert get_meeg_header(ds.pathobj, header) == (None, 0)
    assert "Could not read header of sub-01/eeg/sub-01_task-rest_eeg.vhdr" in caplog.text


def test_get_meeg_header_mne_warnings_are_logged(tmp_path, caplog, recwarn):
    """Warnings raised by MNE while reading a header are logged, not printed."""
    from datalad import api

    ds = api.create(tmp_path / "ds", result_renderer="disabled")
    eeg = ds.pathobj / "sub-01" / "eeg"
    eeg.mkdir(parents=True)
    header = eeg / "sub-01_task-rest_eeg.vhdr"
    header.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        "DataFile=sub-01_task-rest_eeg.eeg\n"
        "MarkerFile=sub-01_task-rest_eye_closed_eeg.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=1\n"
        "SamplingInterval=1000\n"
        "[Binary Infos]\n"
        "BinaryFormat=INT_16\n"
        "[Channel Infos]\n"
        "Ch1=Fp1,,0.1,µV\n"
    )
    header.with_suffix(".vmrk").write_text("Brain Vision Data Exchange Marker File, Version 1.0\n")
    header.with_suffix(".eeg").write_bytes(b"\x00" * 20)
    ds.save(result_renderer="disabled")
    caplog.set_level("WARNING", logger="bids_inventory")

    info, n_samples = get_meeg_header(ds.pathobj, header)

    assert info["sfreq"] == 1000
    # 20 bytes of INT_16 data for 1 channel
    assert n_samples == 10
    assert "MarkerFile 'sub-01_task-rest_eye_closed_eeg.vmrk' not found" in caplog.text
    assert not [w for w in recwarn if "MarkerFile" in str(w.message)]


def test_install_missing_datasets_in_parallel(tmp_path, monkeypatch, capsys, caplog):
    from bids_inventory import _update

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(
        _update,
        "list_openneuro_raw",
        lambda reindex=False: {"ds000001", "ds000002", "ds999999"},
    )
    monkeypatch.setattr(_update, "config_path", lambda *keys: tmp_path)
    monkeypatch.setattr(_update, "superdataset_dir", lambda: None)
    (tmp_path / "ds000002").mkdir()
    caplog.set_level("INFO", logger="bids_inventory")

    _update.install_missing_datasets(jobs=3)

    assert (tmp_path / "ds000001" / "dataset_description.json").exists()
    assert "installed: ds000001" in caplog.text
    assert "already present: ds000002" in caplog.text
    assert "not found: ds999999" in caplog.text
    # datalad output must not mess up the progress bar
    assert "install(ok)" not in capsys.readouterr().out


def test_list_datasets_in_dir_can_resume(tmp_path, monkeypatch):
    """An interrupted update keeps the datasets indexed so far and resumes after them."""
    from bids_inventory import _update
    from bids_inventory.listing import _load_known_datasets

    input_dir = tmp_path / "openneuro"
    for name in ["ds000001", "ds000002", "ds000003"]:
        (input_dir / name).mkdir(parents=True)
    output_file = tmp_path / "openneuro.jsonl"

    indexed = []

    def fake_get_info_dataset(dataset_pth, study_prefix, jobs=1):
        if dataset_pth.name == "ds000003" and "stop" in indexed:
            raise KeyboardInterrupt  # someone stops the update
        indexed.append(dataset_pth.name)
        dataset = _update.new_dataset(dataset_pth.name)
        return dataset

    monkeypatch.setattr(_update, "get_info_dataset", fake_get_info_dataset)
    monkeypatch.setattr(_update, "add_derivatives", lambda dataset, *args: dataset)
    monkeypatch.setattr(_update, "known_derivatives", lambda: [])
    monkeypatch.setattr(_update, "_check_dataset", lambda dataset_pth: True)
    monkeypatch.setattr(_update, "enable_S3_remote", lambda dataset_pth: None)
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)

    # first run is interrupted while indexing ds000003
    indexed.append("stop")
    with pytest.raises(KeyboardInterrupt):
        _update.list_datasets_in_dir(
            _update.init_dataset(), input_dir, debug=False, output_file=output_file
        )
    assert _load_known_datasets(output_file)["name"].tolist() == ["ds000001", "ds000002"]

    # the restart only indexes what is missing
    indexed.clear()
    datasets = _load_known_datasets(output_file).to_dict(orient="list")
    _update.list_datasets_in_dir(datasets, input_dir, debug=False, output_file=output_file)
    assert indexed == ["ds000003"]
    assert _load_known_datasets(output_file)["name"].tolist() == [
        "ds000001",
        "ds000002",
        "ds000003",
    ]
    assert not output_file.with_name("openneuro.jsonl.tmp").exists()


def test_annexed_file_size(tmp_path):
    """The size of a file whose content is not downloaded is read from its annex key."""
    from bids_inventory._update import _annexed_file_size, _brainvision_n_samples

    present = tmp_path / "present.eeg"
    present.write_bytes(b"\x00" * 8)
    assert _annexed_file_size(present) == 8

    not_downloaded = tmp_path / "sub-01_task-rest_eeg.eeg"
    not_downloaded.symlink_to(
        "../.git/annex/objects/q5/pm/SHA256E-s3800016000--5c95f30c.eeg/SHA256E-s3800016000--5c95f30c.eeg"
    )
    assert _annexed_file_size(not_downloaded) == 3800016000
    # ds005486: 66 channels of IEEE_FLOAT_32
    assert _brainvision_n_samples(not_downloaded, "single", 66) == 14394000

    assert _annexed_file_size(tmp_path / "missing.eeg") is None
    assert _brainvision_n_samples(tmp_path / "missing.eeg", "single", 66) is None


@pytest.mark.parametrize(
    "description, expected",
    [
        (None, "n/a"),
        ({}, "n/a"),
        ({"GeneratedBy": [{"Name": "fMRIPrep", "Version": "21.0.1"}]}, "21.0.1"),
        # prefer the entry about the pipeline
        (
            {
                "GeneratedBy": [
                    {"Name": "Manual", "Version": "1"},
                    {"Name": "fMRIPrep", "Version": "23.2.0"},
                ]
            },
            "23.2.0",
        ),
        # BIDS < 1.4
        ({"PipelineDescription": {"Name": "fmriprep", "Version": "1.4.1"}}, "1.4.1"),
        ({"GeneratedBy": [{"Name": "fMRIPrep"}]}, "n/a"),
    ],
)
def test_pipeline_version(description, expected):
    from bids_inventory._update import pipeline_version

    assert pipeline_version(description, "fmriprep") == expected


def test_get_field_strength():
    from bids_inventory._update import get_field_strength

    sidecars = [
        {"MagneticFieldStrength": 3},
        {"MagneticFieldStrength": 3.0},
        {"MagneticFieldStrength": "7T"},
        {"MagneticFieldStrength": "n/a"},
        {"MagneticFieldStrength": 0},
        {"RepetitionTime": 2},
    ]
    assert get_field_strength(sidecars) == [3.0, 7.0]
    assert get_field_strength([]) == []


@pytest.mark.parametrize(
    "version, expected", [("1.8.0", "1.8.0"), ("v1.2.2", "1.2.2"), ("", "n/a")]
)
def test_get_bids_version(tmp_path, version, expected):
    import json

    from bids_inventory._update import get_bids_version

    (tmp_path / "dataset_description.json").write_text(json.dumps({"BIDSVersion": version}))
    assert get_bids_version(tmp_path) == expected
    assert get_bids_version(tmp_path / "missing") == "n/a"


def test_listing_to_dict_adds_new_columns():
    """Listings saved before new columns were added can still be updated."""
    import pandas as pd

    from bids_inventory._update import init_dataset, listing_to_dict

    datasets = listing_to_dict(pd.DataFrame({"name": ["ds000001", "ds000002"]}))
    assert set(datasets) == set(init_dataset())
    assert datasets["bids_version"] == ["n/a", "n/a"]


def test_list_datasets_in_dir_reindex_can_resume(tmp_path, monkeypatch):
    """Re-indexing replaces entries in place and can be resumed."""
    import pandas as pd

    from bids_inventory import _update
    from bids_inventory.listing import _load_known_datasets, save_known_datasets

    input_dir = tmp_path / "openneuro"
    for name in ["ds000001", "ds000002"]:
        (input_dir / name).mkdir(parents=True)
    output_file = tmp_path / "openneuro.jsonl"

    # listing indexed before a new field was added
    old = _update.init_dataset()
    for name in ["ds000001", "ds000002"]:
        _update._add_or_replace(old, _update.new_dataset(name))
    save_known_datasets(pd.DataFrame.from_dict(old), output_file)

    indexed = []

    def fake_get_info_dataset(dataset_pth, study_prefix, jobs=1):
        if dataset_pth.name == "ds000002" and "stop" in indexed:
            raise KeyboardInterrupt
        indexed.append(dataset_pth.name)
        dataset = _update.new_dataset(dataset_pth.name)
        dataset.bids_version = "1.10.0"
        return dataset

    monkeypatch.setattr(_update, "get_info_dataset", fake_get_info_dataset)
    monkeypatch.setattr(_update, "add_derivatives", lambda dataset, *args: dataset)
    monkeypatch.setattr(_update, "known_derivatives", lambda: [])
    monkeypatch.setattr(_update, "_check_dataset", lambda dataset_pth: True)
    monkeypatch.setattr(_update, "enable_S3_remote", lambda dataset_pth: None)
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)

    def run():
        datasets = _update.listing_to_dict(_load_known_datasets(output_file))
        return _update.list_datasets_in_dir(
            datasets, input_dir, debug=False, output_file=output_file, reindex=True
        )

    # interrupted while re-indexing ds000002
    indexed.append("stop")
    with pytest.raises(KeyboardInterrupt):
        run()
    listing = _load_known_datasets(output_file).set_index("name")
    # the listing stays complete: ds000002 keeps its previous entry
    assert listing.index.tolist() == ["ds000001", "ds000002"]
    assert listing.loc["ds000001", "bids_version"] == "1.10.0"
    assert pd.isna(listing.loc["ds000002", "bids_version"])

    # the restart only re-indexes ds000002
    indexed.clear()
    run()
    assert indexed == ["ds000002"]
    listing = _load_known_datasets(output_file).set_index("name")
    assert listing["bids_version"].tolist() == ["1.10.0", "1.10.0"]
    assert _update._read_reindex_progress(_update.reindex_progress_file(output_file)) == {
        "ds000001",
        "ds000002",
    }


def test_skipped_datasets_are_not_indexed(tmp_path, monkeypatch):
    from bids_inventory import _update

    input_dir = tmp_path / "openneuro"
    for name in ["ds000001", "ds000002"]:
        (input_dir / name).mkdir(parents=True)

    indexed = []

    def fake_get_info_dataset(dataset_pth, study_prefix, jobs=1):
        indexed.append(dataset_pth.name)
        dataset = _update.new_dataset(dataset_pth.name)
        return dataset

    monkeypatch.setattr(_update, "get_info_dataset", fake_get_info_dataset)
    monkeypatch.setattr(_update, "add_derivatives", lambda dataset, *args: dataset)
    monkeypatch.setattr(_update, "known_derivatives", lambda: [])
    monkeypatch.setattr(_update, "_check_dataset", lambda dataset_pth: True)
    monkeypatch.setattr(_update, "enable_S3_remote", lambda dataset_pth: None)
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)
    monkeypatch.setattr(_update, "skipped_datasets", lambda: {"ds000002": "deleted"})

    datasets = _update.list_datasets_in_dir(_update.init_dataset(), input_dir, debug=False)

    assert indexed == ["ds000001"]
    assert datasets["name"] == ["ds000001"]


def test_get_scan_duration_error_reason(tmp_path, caplog):
    """The error says why the header could not be read."""
    from datalad import api

    from bids_inventory._update import get_scan_duration

    ds = api.create(tmp_path / "ds", result_renderer="disabled")
    bold = ds.pathobj / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz"
    bold.parent.mkdir(parents=True)
    bold.write_bytes(b"not really a nifti file")
    ds.save(result_renderer="disabled")
    # the content is not available anywhere anymore
    ds.drop(bold, reckless="kill", result_renderer="disabled")

    n_volumes, repetition_time = get_scan_duration(ds.pathobj, bold)
    assert n_volumes == 0
    assert math.isnan(repetition_time)
    assert "Could not get duration for: sub-01/func/sub-01_task-rest_bold.nii.gz (" in caplog.text
    assert "Could not find a usable URL" in caplog.text


def test_get_scan_duration_retries_once(monkeypatch, tmp_path):
    from bids_inventory import _update

    calls = []

    def flaky(dataset_pth, filepath):
        calls.append(filepath)
        return (None, "temporary error") if len(calls) == 1 else ((180, 2.0), "")

    monkeypatch.setattr(_update, "_read_nifti_header", flaky)
    bold = tmp_path / "sub-01_task-rest_bold.nii.gz"
    assert _update.get_scan_duration(tmp_path, bold) == (180, 2.0)
    assert len(calls) == 2


def test_update_non_openneuro(tmp_path, monkeypatch, caplog):
    """Datasets not on OpenNeuro are indexed with a prefix, and only once."""
    from bids_inventory import _update
    from bids_inventory.listing import _load_known_datasets

    base_dir = tmp_path / "non_openneuro"
    for folder in [
        "abide/Caltech",
        "abide/NYU",
        "cneuromod/friends",
        "cneuromod/code",
        "dandi/000108",
        "dandi/not_listed",
    ]:
        (base_dir / folder).mkdir(parents=True)
    listing = tmp_path / "non_openneuro.jsonl"
    listing.write_text("")

    indexed = []

    def fake_get_info_dataset(dataset_pth, study_prefix, jobs=1):
        indexed.append(f"{study_prefix}{dataset_pth.name}")
        return _update.new_dataset(f"{study_prefix}{dataset_pth.name}")

    monkeypatch.setattr(_update, "get_info_dataset", fake_get_info_dataset)
    monkeypatch.setattr(_update, "add_derivatives", lambda dataset, *args: dataset)
    # not needed for the datasets that are not on OpenNeuro
    monkeypatch.setattr(_update, "known_derivatives", lambda: pytest.fail("not needed"))
    monkeypatch.setattr(_update, "_check_dataset", lambda dataset_pth: True)
    monkeypatch.setattr(_update, "enable_S3_remote", lambda dataset_pth: None)
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)
    monkeypatch.setattr(_update, "config_path", lambda *keys: base_dir)
    monkeypatch.setattr(_update, "_non_openneuro_listing", lambda: listing)
    monkeypatch.setattr(
        _update, "_load_known_datasets", lambda f: _load_or_empty(_load_known_datasets, f)
    )
    installs = []
    monkeypatch.setattr(
        _update,
        "install_non_openneuro",
        lambda base_dir, collections, jobs: installs.append(sorted(collections)),
    )

    _update.update(debug=False, source="non-openneuro")

    # the missing datasets of non_openneuro_datasets.yml are installed first
    assert installs[0] == sorted(_update.non_openneuro_collections())
    # only the cneuromod subdatasets included, and the dandi datasets listed, are indexed
    assert indexed == ["ABIDE_Caltech", "ABIDE_NYU", "CNEUROMOD_friends", "DANDI_000108"]
    assert _load_known_datasets(listing)["name"].tolist() == indexed
    # missing collections are skipped with a warning
    assert "corr not found" in caplog.text

    # a second run does not index them again
    indexed.clear()
    _update.update(debug=False, source="non-openneuro")
    assert indexed == []


def _load_or_empty(load, listing):
    """Load a listing, or return an empty one if the file is empty."""
    import pandas as pd

    from bids_inventory._update import init_dataset

    if listing.stat().st_size == 0:
        return pd.DataFrame(init_dataset())
    return load(listing)


@pytest.mark.parametrize(
    "stdout, stderr, expected",
    [
        (
            "",
            "[INFO] Attempting a clone\n"
            "[ERROR] could not create work tree dir: No space left on device\n"
            "[INFO] Completed clone attempts for Dataset(/data/ds000001)\n",
            "could not create work tree dir: No space left on device",
        ),
        (
            "install(error): /data/ds000001 (dataset) [Failed to clone from any source]\n",
            "[INFO] Completed clone attempts for Dataset(/data/ds000001)\n",
            "install(error): /data/ds000001 (dataset) [Failed to clone from any source]",
        ),
        (
            "",
            "[INFO] init: 1 failed\ngit-annex: fd:19: Data.ByteString.hGetLine: end of file\n",
            "git-annex: fd:19: Data.ByteString.hGetLine: end of file",
        ),
        ("", "", ""),
    ],
)
def test_error_reason(stdout, stderr, expected):
    """The reason of a failure is the error, not datalad's last progress message."""
    from bids_inventory._update import _error_reason

    assert _error_reason(stdout, stderr) == expected


@pytest.fixture
def superdataset(tmp_path, monkeypatch):
    """Superdataset with ds000001 and ds000002 installed, ds000003 not installed."""
    from bids_inventory._update import CONFIG_ENV_VAR, config

    path = tmp_path / "datalad" / "openneuro"
    for name in ["ds000001", "ds000002"]:
        (path / name / ".git").mkdir(parents=True)
    (path / "ds000003").mkdir()
    install_dir = tmp_path / "openneuro"
    install_dir.mkdir()
    derivatives = tmp_path / "datalad" / "openneuro-derivatives"
    (derivatives / ".git").mkdir(parents=True)

    user_config = tmp_path / "config.yml"
    user_config.write_text(
        "local_paths:\n"
        "    datalad:\n"
        f"        OpenNeuroDatasets: {path}\n"
        f"        OpenNeuroDerivatives: {derivatives}\n"
        f"    openneuro:\n        OpenNeuroDatasets: {install_dir}\n"
    )
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(user_config))
    config.cache_clear()
    yield path, install_dir
    config.cache_clear()


def test_superdataset_is_required(tmp_path, monkeypatch):
    from bids_inventory._update import CONFIG_ENV_VAR, config, superdataset_dir

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    config.cache_clear()
    with pytest.raises(ValueError, match="'local_paths.datalad.OpenNeuroDatasets' is not set"):
        superdataset_dir()
    config.cache_clear()


def test_install_missing_datasets_uses_superdataset(superdataset, monkeypatch):
    """Datasets of the superdataset are installed there, the others in the install folder."""
    from bids_inventory import _update

    path, install_dir = superdataset
    # ds000001: installed in the superdataset, ds000003: in the superdataset, not installed,
    # ds000005: not in the superdataset
    monkeypatch.setattr(
        _update, "list_openneuro_raw", lambda reindex=False: {"ds000001", "ds000003", "ds000005"}
    )
    in_superdataset, elsewhere = [], []
    monkeypatch.setattr(
        _update,
        "_install_in_superdataset",
        lambda sds, names, jobs=1: in_superdataset.append((sds, names)),
    )
    monkeypatch.setattr(
        _update,
        "_install_openneuro_dataset",
        lambda data_pth: elsewhere.append(data_pth) or ("installed", ""),
    )

    _update.install_missing_datasets()

    assert in_superdataset == [(path, ["ds000003"])]
    assert elsewhere == [install_dir / "ds000005"]


def test_install_in_superdataset(tmp_path):
    """Datasets registered in the superdataset but not installed get installed in it."""
    from datalad import api

    from bids_inventory._update import _install_in_superdataset

    super_ds = api.create(tmp_path / "super", result_renderer="disabled")
    for name in ["ds000001", "ds000002"]:
        api.create(super_ds.pathobj / name, dataset=super_ds, result_renderer="disabled")
    clone = api.clone(super_ds.path, tmp_path / "clone", result_renderer="disabled")
    assert not (clone.pathobj / "ds000001" / ".git").exists()

    _install_in_superdataset(clone.pathobj, ["ds000001", "ds000002"], jobs=2)

    assert (clone.pathobj / "ds000001" / ".git").exists()
    assert (clone.pathobj / "ds000002" / ".git").exists()


def test_list_openneuro_derivatives(superdataset, tmp_path, monkeypatch):
    """Derivatives of the superdataset first, then those only on GitHub."""
    import pandas as pd

    from bids_inventory import _update

    derivatives = tmp_path / "datalad" / "openneuro-derivatives"
    (derivatives / "ds000001-fmriprep").mkdir()
    monkeypatch.setattr(
        _update, "get_list_of_datasets", lambda gh_orga: ["ds000001-fmriprep", "ds000002-mriqc"]
    )
    monkeypatch.setattr(_update, "_data_dir", lambda: tmp_path)

    _update.list_openneuro_derivatives()

    listed = pd.read_csv(tmp_path / "OpenNeuroDerivatives.tsv", sep="\t")["name"].tolist()
    assert listed == ["ds000001-fmriprep", "ds000002-mriqc"]


def test_derivative_description_from_superdataset(superdataset, tmp_path, monkeypatch):
    """The version of a derivative is read from the superdataset if it is installed there."""
    import json

    from bids_inventory import _update

    derivatives = tmp_path / "datalad" / "openneuro-derivatives"
    installed = derivatives / "ds000001-fmriprep"
    (installed / ".git").mkdir(parents=True)
    (installed / "dataset_description.json").write_text(
        json.dumps({"GeneratedBy": [{"Name": "fMRIPrep", "Version": "23.0.0"}]})
    )
    (derivatives / "ds000002-fmriprep").mkdir()  # not installed
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)
    remote = []
    monkeypatch.setattr(
        _update, "_remote_dataset_description", lambda url: remote.append(url) or None
    )

    description = _update._derivative_description("ds000001-fmriprep")
    assert _update.pipeline_version(description, "fmriprep") == "23.0.0"
    assert remote == []

    _update._derivative_description("ds000002-fmriprep")
    assert remote == [
        "https://raw.githubusercontent.com/OpenNeuroDerivatives/"
        "ds000002-fmriprep/main/dataset_description.json"
    ]


def test_update_openneuro_indexes_superdataset(superdataset, tmp_path, monkeypatch):
    """Datasets are indexed from the superdataset if they are installed there."""
    from bids_inventory import _update
    from bids_inventory.listing import _load_known_datasets

    path, install_dir = superdataset
    for name in ["ds000002", "ds000004"]:
        (install_dir / name).mkdir()
    listing = tmp_path / "openneuro.jsonl"
    listing.write_text("")
    (tmp_path / "OpenNeuroDatasets.tsv").write_text("name\nds000001\n")

    indexed = []

    def fake_get_info_dataset(dataset_pth, study_prefix, jobs=1):
        indexed.append(str(dataset_pth.relative_to(tmp_path)))
        return _update.new_dataset(dataset_pth.name)

    monkeypatch.setattr(_update, "get_info_dataset", fake_get_info_dataset)
    monkeypatch.setattr(_update, "add_derivatives", lambda dataset, *args: dataset)
    monkeypatch.setattr(_update, "known_derivatives", lambda: [])
    monkeypatch.setattr(_update, "_check_dataset", lambda dataset_pth: True)
    monkeypatch.setattr(_update, "enable_S3_remote", lambda dataset_pth: None)
    monkeypatch.setattr(_update, "_get_file", lambda ds, filepath: None)
    monkeypatch.setattr(_update, "skipped_datasets", lambda: {})
    monkeypatch.setattr(_update, "_data_dir", lambda: tmp_path)
    monkeypatch.setattr(_update, "_openneuro_listing", lambda: listing)
    monkeypatch.setattr(
        _update, "_load_known_datasets", lambda f: _load_or_empty(_load_known_datasets, f)
    )

    _update.update_openneuro(debug=False)

    # ds000002 is in both folders: only indexed from the superdataset
    assert indexed == [
        "datalad/openneuro/ds000001",
        "datalad/openneuro/ds000002",
        "openneuro/ds000004",
    ]
    assert _load_known_datasets(listing)["name"].tolist() == ["ds000001", "ds000002", "ds000004"]


@pytest.mark.parametrize(
    "output, expected",
    [
        ("3235 annex'd files (0.0 B/54.0 GB present/total size)\nnothing to save", "54.0 GB"),
        # other files can be listed before the summary
        (
            "untracked: /data/ds003816/ses-01 (directory)\n"
            " modified: config (file)\n"
            "3235 annex'd files (0.0 B/54.0 GB present/total size)\n",
            "54.0 GB",
        ),
        ("12 annex'd files (1.2 MB/740.5 KB present/total size)", "740.5 KB"),
        ("nothing to save, working tree clean", "n/a"),
    ],
)
def test_parse_dataset_size(output, expected):
    from bids_inventory._update import parse_dataset_size

    assert parse_dataset_size(output) == expected


def test_get_list_of_datasets_ignores_empty_repositories(monkeypatch):
    """OpenNeuro creates empty repositories for datasets that are not public."""
    from bids_inventory import _update

    pages = [
        [
            {"name": "ds000001", "size": 331},
            {"name": "ds008759", "size": 0},
            {"name": ".github", "size": 5},
        ],
        [{"name": "ds000002", "size": 12}],
        [],
    ]
    monkeypatch.setattr(_update, "get_auth", lambda: None)
    monkeypatch.setattr(
        _update, "request_list_of_repos", lambda gh_orga, page, auth=None: pages[page - 1]
    )

    assert _update.get_list_of_datasets("OpenNeuroDatasets") == ["ds000001", "ds000002"]


def test_check_dataset_skips_empty_repository(tmp_path, monkeypatch):
    import subprocess

    from bids_inventory import _update

    empty = tmp_path / "ds008759"
    subprocess.run(["git", "init", "-q", str(empty)], check=True)
    assert _update._is_empty_repository(empty)
    assert not _update._is_empty_repository(tmp_path / "not_a_repo")

    def no_network(*args, **kwargs):
        raise AssertionError("an empty repository should be skipped without a request")

    monkeypatch.setattr(_update.requests, "get", no_network)
    assert _update._check_dataset(empty) is False

    (empty / "README").write_text("dataset")
    subprocess.run(["git", "-C", str(empty), "add", "README"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(empty),
            "-c",
            "user.name=a",
            "-c",
            "user.email=a@b.c",
            "commit",
            "-qm",
            "init",
        ],
        check=True,
    )
    assert not _update._is_empty_repository(empty)


def test_get_raw_url_non_openneuro(tmp_path):
    """The URL of datasets that are not on OpenNeuro is read without printing anything."""
    import subprocess
    import sys

    from datalad import api

    source = api.create(tmp_path / "source", result_renderer="disabled")
    api.clone(source.path, tmp_path / "Leuven_2", result_renderer="disabled")

    # in a separate process: datalad does not print its results the same way under pytest
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from bids_inventory._update import get_raw_url; "
            "url = get_raw_url(Path(sys.argv[1])); print('URL', url)",
            str(tmp_path / "Leuven_2"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines() == [f"URL {source.path}"]


def test_get_scan_durations_no_usable_url(tmp_path, monkeypatch, caplog):
    """If the first run has no public URL, the other runs of the dataset are not tried."""
    import math

    from bids_inventory import _update

    calls = []

    def no_url(dataset_pth, filepath):
        calls.append(filepath.name)
        return None, f"Could not find a usable URL for {filepath.relative_to(dataset_pth)}"

    monkeypatch.setattr(_update, "_read_nifti_header", no_url)
    files = [
        tmp_path / "sub-01" / "func" / f"sub-01_task-rest_run-{i}_bold.nii.gz" for i in (1, 2, 3)
    ]

    durations = _update.get_scan_durations(tmp_path, files, jobs=2)

    # the first run is tried twice, the others not at all
    assert calls == [files[0].name, files[0].name]
    assert all(n == 0 and math.isnan(t) for n, t in durations.values())
    assert set(durations) == set(files)
    assert caplog.text.count("Could not get duration") == 1
    assert "the 3 runs" in caplog.text


def test_get_scan_durations_other_errors(tmp_path, monkeypatch):
    """Other errors on the first run do not prevent reading the others."""
    from bids_inventory import _update

    def flaky(dataset_pth, filepath):
        if "run-1" in filepath.name:
            return None, "fsspec.exceptions.FSTimeoutError"
        return (180, 2.0), ""

    monkeypatch.setattr(_update, "_read_nifti_header", flaky)
    files = [tmp_path / f"sub-01_task-rest_run-{i}_bold.nii.gz" for i in (1, 2)]

    durations = _update.get_scan_durations(tmp_path, files)

    assert durations[files[0]][0] == 0
    assert durations[files[1]] == (180, 2.0)


def _touch(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_runs_one_file_per_run(tmp_path):
    """BrainVision runs are represented by their .vhdr, sidecars and companions are left out."""
    from bids_inventory._update import _runs

    eeg = tmp_path / "sub-01" / "eeg"
    files = [
        _touch(eeg / f"sub-01_task-a_run-1_eeg{ext}") for ext in [".eeg", ".vhdr", ".vmrk", ".json"]
    ]
    files += [_touch(eeg / f"sub-01_task-a_run-2_eeg{ext}") for ext in [".set", ".fdt"]]

    assert [x.name for x in _runs(files, "eeg")] == [
        "sub-01_task-a_run-1_eeg.vhdr",
        "sub-01_task-a_run-2_eeg.set",
    ]


def test_sidecar_duration_inheritance(tmp_path):
    """The closest sidecar whose entities are all in the run is used."""
    import json

    from bids_inventory._update import _sidecar_duration

    meg = tmp_path / "sub-01" / "ses-1" / "meg"
    run_1 = _touch(meg / "sub-01_ses-1_task-a_run-1_meg.fif")
    run_2 = _touch(meg / "sub-01_ses-1_task-a_run-2_meg.fif")
    other_task = _touch(meg / "sub-01_ses-1_task-b_meg.fif")
    _touch(meg / "sub-01_ses-1_task-a_run-1_meg.json", json.dumps({"RecordingDuration": 300}))
    # at the root: for all the runs of task a
    _touch(tmp_path / "task-a_meg.json", json.dumps({"RecordingDuration": 600.5}))

    assert _sidecar_duration(tmp_path, run_1) == 300
    assert _sidecar_duration(tmp_path, run_2) == 600.5
    assert _sidecar_duration(tmp_path, other_task) is None


def test_sidecar_duration_pet_and_func(tmp_path):
    import json

    from bids_inventory._update import _sidecar_duration

    pet = _touch(tmp_path / "sub-01" / "pet" / "sub-01_pet.nii.gz")
    _touch(pet.with_name("sub-01_pet.json"), json.dumps({"FrameDuration": [60, 60, 120]}))
    assert _sidecar_duration(tmp_path, pet) == 240

    # func durations are read from the NIfTI headers
    bold = _touch(tmp_path / "sub-01" / "func" / "sub-01_task-a_bold.nii.gz")
    _touch(bold.with_name("sub-01_task-a_bold.json"), json.dumps({"RecordingDuration": 10}))
    assert _sidecar_duration(tmp_path, bold) is None

    # a malformed sidecar is not an error
    nirs = _touch(tmp_path / "sub-01" / "nirs" / "sub-01_task-a_nirs.snirf")
    _touch(nirs.with_name("sub-01_task-a_nirs.json"), "{not json")
    assert _sidecar_duration(tmp_path, nirs) is None


def test_get_duration_from_sidecars(tmp_path, monkeypatch):
    """The durations of MEG, NIRS and motion runs are read from their sidecars."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01"])
    for datatype, ext in [("meg", ".fif"), ("nirs", ".snirf"), ("motion", ".tsv")]:
        run = _touch(tmp_path / "sub-01" / datatype / f"sub-01_task-a_{datatype}{ext}")
        _touch(
            run.with_name(f"sub-01_task-a_{datatype}.json"), json.dumps({"RecordingDuration": 60})
        )

    assert _update.get_duration(tmp_path, ["meg", "nirs", "motion"], ["a"]) == {
        datatype: {"a": [(1, 60.0)]} for datatype in ["meg", "nirs", "motion"]
    }


def test_update_durations(tmp_path, monkeypatch):
    """Only the durations of the requested datatypes of the installed datasets change."""
    import pandas as pd

    from bids_inventory import _update

    listing = pd.DataFrame(
        {
            "name": ["ds000001", "ds000002", "ds000003"],
            "datatypes": [["func", "meg"], ["meg"], ["anat"]],
            "tasks": [["a"], ["b"], []],
            "duration": [{"func": {"a": [360.0]}, "meg": {"a": None}}, {"meg": {"b": None}}, {}],
        }
    )
    saved = {}
    monkeypatch.setattr(_update, "_load_known_datasets", lambda _: listing.copy())
    monkeypatch.setattr(
        _update, "save_known_datasets", lambda df, output_file: saved.update(df=df.copy())
    )
    # ds000002 is not installed
    monkeypatch.setattr(
        _update,
        "_local_openneuro_dataset",
        lambda name: tmp_path / name if name == "ds000001" else None,
    )
    monkeypatch.setattr(
        _update,
        "get_duration",
        lambda dataset_pth, datatypes, tasks, jobs: {"meg": {"a": [(1, 120.0)]}},
    )
    monkeypatch.setattr(
        _update, "get_nb_channels", lambda dataset_pth, datatypes, tasks: {"meg": [306]}
    )

    _update.update_durations(["meg"], debug=False)

    assert saved["df"]["duration"].tolist() == [
        {"func": {"a": [360.0]}, "meg": {"a": [(1, 120.0)]}},
        {"meg": {"b": None}},
        {},
    ]
    assert saved["df"]["nb_channels"].tolist() == [{"meg": [306]}, None, None]


def test_get_duration_pet_without_task(tmp_path, monkeypatch):
    """PET runs without task entity are listed together."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01"])
    for session in ["baseline", "rescan"]:
        run = _touch(
            tmp_path / "sub-01" / f"ses-{session}" / "pet" / f"sub-01_ses-{session}_pet.nii.gz"
        )
        _touch(
            run.with_name(f"sub-01_ses-{session}_pet.json"), json.dumps({"FrameDuration": [60, 30]})
        )

    assert _update.get_duration(tmp_path, ["anat", "pet"], []) == {"pet": [(1, 90.0), (1, 90.0)]}


def test_get_nb_channels(tmp_path, monkeypatch):
    """The number of channels of each run is read from the sidecars."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01"])
    eeg = _touch(tmp_path / "sub-01" / "eeg" / "sub-01_task-a_eeg.edf")
    _touch(eeg.with_name("sub-01_task-a_eeg.json"), json.dumps({"EEGChannelCount": 64}))
    ieeg = _touch(tmp_path / "sub-01" / "ieeg" / "sub-01_task-a_ieeg.edf")
    _touch(
        ieeg.with_name("sub-01_task-a_ieeg.json"),
        json.dumps({"ECOGChannelCount": 20, "SEEGChannelCount": 10}),
    )
    # inherited from the root, for all the runs of the task
    for run in [1, 2]:
        _touch(tmp_path / "sub-01" / "meg" / f"sub-01_task-a_run-{run}_meg.fif")
    _touch(tmp_path / "task-a_meg.json", json.dumps({"MEGChannelCount": 306}))
    # no channel count, and not a datatype with channels
    _touch(tmp_path / "sub-01" / "nirs" / "sub-01_task-a_nirs.snirf")

    assert _update.get_nb_channels(tmp_path, ["eeg", "ieeg", "meg", "nirs"], ["a"]) == {
        "eeg": [64],
        "ieeg": [30],
        "meg": [306, 306],
    }


def test_get_readme_and_bidsignore_size(tmp_path):
    import os

    from bids_inventory._update import get_bidsignore_size, get_readme_size

    assert get_readme_size(tmp_path) is None
    assert get_bidsignore_size(tmp_path) is None

    (tmp_path / "README.md").write_text("# a dataset\n")
    assert get_readme_size(tmp_path) == 12
    (tmp_path / ".bidsignore").write_text("extra/\n")
    assert get_bidsignore_size(tmp_path) == 7

    # annexed README whose content is not downloaded: size from its git-annex key
    (tmp_path / "README.md").unlink()
    os.symlink(".git/annex/objects/xx/MD5E-s1234--abcd/MD5E-s1234--abcd", tmp_path / "README")
    assert get_readme_size(tmp_path) == 1234


def test_update_fields(tmp_path, monkeypatch):
    """Only the requested fields of the installed datasets change, sizes stay integers."""
    import pandas as pd

    from bids_inventory import _update

    listing = pd.DataFrame({"name": ["ds000001", "ds000002"], "readme_size": [None, 10]})
    saved = {}
    monkeypatch.setattr(_update, "_load_known_datasets", lambda _: listing.copy())
    monkeypatch.setattr(
        _update, "save_known_datasets", lambda df, output_file: saved.update(df=df.copy())
    )
    monkeypatch.setattr(
        _update,
        "_local_openneuro_dataset",
        lambda name: tmp_path if name == "ds000001" else None,
    )
    (tmp_path / "README").write_text("hello")

    _update.update_fields(["readme_size", "bidsignore_size"], debug=False)

    assert saved["df"]["readme_size"].tolist() == [5, 10]
    assert saved["df"]["bidsignore_size"].tolist() == [None, None]

    with pytest.raises(ValueError, match="Unknown fields"):
        _update.update_fields(["foo"], debug=False)


@pytest.mark.parametrize(
    "description, expected",
    [
        ({"HEDVersion": "8.2.0", "Keywords": ["EEG", " vision "]}, (["8.2.0"], ["EEG", "vision"])),
        ({"HEDVersion": ["8.2.0", "sc:score_1.0.0"]}, (["8.2.0", "sc:score_1.0.0"], [])),
        # empty values and placeholders
        ({"HEDVersion": "", "Keywords": ["n/a", None, ""]}, ([], [])),
        ({}, ([], [])),
    ],
)
def test_get_hed_version_and_keywords(tmp_path, description, expected):
    import json

    from bids_inventory._update import get_hed_version, get_keywords

    (tmp_path / "dataset_description.json").write_text(json.dumps(description))
    assert (get_hed_version(tmp_path), get_keywords(tmp_path)) == expected


def test_get_references_and_links_invalid_description(tmp_path):
    """An empty dataset_description.json is not an error."""
    from bids_inventory._update import get_references_and_links

    (tmp_path / "dataset_description.json").write_text("")
    assert get_references_and_links(tmp_path) == []


def test_local_openneuro_dataset_most_recent_copy(tmp_path, monkeypatch):
    """A dataset installed twice is read from the copy with the most recent commit."""
    from bids_inventory import _update

    superdataset, other = tmp_path / "superdataset", tmp_path / "other"
    for folder in [superdataset, other]:
        _touch(folder / "ds000001" / "dataset_description.json", "{}")
    _touch(superdataset / "ds000002" / "dataset_description.json", "{}")
    monkeypatch.setattr(_update, "superdataset_dir", lambda: superdataset)
    monkeypatch.setattr(_update, "config_path", lambda *keys: other)
    times = {superdataset / "ds000001": 100, other / "ds000001": 200}
    monkeypatch.setattr(_update, "_last_commit_time", lambda path: times[path])

    assert _update._local_openneuro_dataset("ds000001") == other / "ds000001"
    assert _update._local_openneuro_dataset("ds000002") == superdataset / "ds000002"
    assert _update._local_openneuro_dataset("ds000003") is None


def test_get_task_ids(tmp_path, monkeypatch):
    """The CogAtlasID of each task, from the sidecars of its runs (with inheritance)."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01"])
    func = tmp_path / "sub-01" / "func"
    for run in [1, 2]:
        _touch(func / f"sub-01_task-rest_run-{run}_bold.nii.gz")
        # not runs: their sidecars do not describe the task
        _touch(func / f"sub-01_task-rest_run-{run}_events.tsv")
    _touch(tmp_path / "task-rest_bold.json", json.dumps({"CogAtlasID": "trm_4c8a834779883"}))
    _touch(tmp_path / "task-rest_events.json", json.dumps({"CogAtlasID": "not a run"}))
    eeg = _touch(tmp_path / "sub-01" / "eeg" / "sub-01_task-nback_eeg.edf")
    _touch(
        eeg.with_name("sub-01_task-nback_eeg.json"),
        json.dumps(
            {"CogAtlasID": ["tsk_4a57abb949e1a", "https://www.cognitiveatlas.org/fillYourID"]}
        ),
    )
    # no sidecar, or no suffix
    _touch(tmp_path / "sub-01" / "beh" / "sub-01_task-other_beh.tsv")
    _touch(tmp_path / "sub-01" / "beh" / "sub-01_task-other.tsv")

    assert _update.get_task_ids(tmp_path, "CogAtlasID") == {
        "nback": ["tsk_4a57abb949e1a"],
        "rest": ["trm_4c8a834779883"],
    }
    assert _update.get_task_ids(tmp_path, "CogPOID") == {}


def test_non_openneuro_collections():
    """Each collection has a unique prefix and datasets, and a source in the dashboard."""
    import pandas as pd

    from bids_inventory._update import non_openneuro_collections
    from bids_inventory.wrangling import _get_source_study

    collections = non_openneuro_collections()
    prefixes = [settings["prefix"] for settings in collections.values()]
    assert len(set(prefixes)) == len(prefixes)
    for collection, settings in collections.items():
        assert settings["prefix"].endswith("_"), collection
        assert bool(settings.get("superdataset")) != bool(settings.get("datasets")), collection
    sources = _get_source_study(pd.DataFrame({"name": [f"{x}dataset" for x in prefixes]}))
    assert "other" not in sources


def test_install_non_openneuro(tmp_path, monkeypatch):
    """Only the missing datasets are installed, with their root files."""
    import subprocess
    from pathlib import Path

    from bids_inventory import _update

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command[1:])
        if command[1] == "install":
            # datalad install -s <source> <path>
            (Path(command[-1]) / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    root_files = []
    monkeypatch.setattr(_update.subprocess, "run", fake_run)
    monkeypatch.setattr(_update, "_get_root_files", lambda path: root_files.append(path.name))
    (tmp_path / "nemar" / "nm000103" / ".git").mkdir(parents=True)

    _update.install_non_openneuro(
        tmp_path,
        {
            "nemar": {
                "prefix": "NEMAR_",
                "datasets": {
                    "nm000103": "https://x/nm000103.git",
                    "nm000104": "https://x/nm000104.git",
                },
            }
        },
    )

    # nm000103 is already installed
    assert commands == [
        ["install", "-s", "https://x/nm000104.git", str(tmp_path / "nemar" / "nm000104")]
    ]
    assert root_files == ["nm000104"]


def test_install_non_openneuro_error(tmp_path, monkeypatch, caplog):
    """A dataset that cannot be installed is skipped with an error."""
    import subprocess

    from bids_inventory import _update

    monkeypatch.setattr(
        _update.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "[ERROR] Failed to connect to datasets.datalad.org"
        ),
    )
    _update.install_non_openneuro(
        tmp_path, {"indi": {"prefix": "INDI_", "datasets": {"HypnosisBarrios": "///indi/x"}}}
    )
    assert "Could not install ///indi/x: Failed to connect to datasets.datalad.org" in caplog.text


def test_update_bids_datatypes(tmp_path, monkeypatch):
    """The datatypes are read from the BIDS schema, and kept if it cannot be downloaded."""
    import json

    import requests

    from bids_inventory import _update

    datatypes_file = tmp_path / "bids_datatypes.json"
    monkeypatch.setattr(_update, "_bids_datatypes_file", lambda: datatypes_file)
    monkeypatch.setattr(_update, "KNOWN_DATATYPES", ["anat"])

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "bids_version": "1.11.2",
                "schema_version": "2.0.0",
                "objects": {
                    "datatypes": {
                        "emg": {"value": "emg", "display_name": "Electromyography"},
                        "anat": {
                            "value": "anat",
                            "display_name": "Anatomical Magnetic Resonance Imaging",
                        },
                    }
                },
            }

    monkeypatch.setattr(_update.requests, "get", lambda url, timeout: Response())
    _update.update_bids_datatypes()

    saved = json.loads(datatypes_file.read_text())
    assert saved["bids_version"] == "1.11.2"
    assert list(saved["datatypes"]) == ["anat", "emg"]
    assert _update.KNOWN_DATATYPES == ["anat", "emg"]

    def no_network(url, timeout):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(_update.requests, "get", no_network)
    _update.update_bids_datatypes()
    assert json.loads(datatypes_file.read_text()) == saved


def test_known_datatypes():
    """The official BIDS datatypes, saved from the BIDS schema."""
    from bids_inventory.listing import KNOWN_DATATYPES

    assert {"anat", "func", "eeg", "emg", "micr", "nirs", "motion"} <= set(KNOWN_DATATYPES)


def test_get_physio_recordings(tmp_path):
    from bids_inventory._update import get_physio_recordings

    func = tmp_path / "sub-01" / "ses-1" / "func"
    for name in [
        "sub-01_ses-1_task-visualSearch_recording-eye1_physio.tsv.gz",
        "sub-01_ses-1_task-visualSearch_recording-eye1_physio.json",
        "sub-01_ses-1_task-rest_run-1_recording-cardiac_physio.tsv.gz",
        # no recording entity
        "sub-01_ses-1_task-rest_run-2_physio.tsv.gz",
    ]:
        _touch(func / name)

    assert get_physio_recordings(tmp_path) == ["cardiac", "eye1"]


def test_local_non_openneuro_dataset(tmp_path, monkeypatch):
    from bids_inventory import _update

    (tmp_path / "abide" / "Caltech").mkdir(parents=True)
    (tmp_path / "abide2" / "Caltech").mkdir(parents=True)
    monkeypatch.setattr(_update, "config_path", lambda *keys: tmp_path)

    assert _update._local_non_openneuro_dataset("ABIDE_Caltech") == tmp_path / "abide" / "Caltech"
    assert _update._local_non_openneuro_dataset("ABIDE2_Caltech") == tmp_path / "abide2" / "Caltech"
    assert _update._local_non_openneuro_dataset("ABIDE_NYU") is None
    assert _update._local_non_openneuro_dataset("ds000001") is None


def test_get_physio_columns(tmp_path, monkeypatch):
    """From the sidecars of the physio files of the first participant (with inheritance)."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01"])
    func = tmp_path / "sub-01" / "func"
    _touch(func / "sub-01_task-rest_physio.tsv.gz")
    _touch(func / "sub-01_task-rest_physio.json", json.dumps({"Columns": ["cardiac", "trigger"]}))
    _touch(func / "sub-01_task-nback_physio.tsv.gz")
    # inherited, and a single column as a string
    _touch(tmp_path / "task-nback_physio.json", json.dumps({"Columns": "respiratory"}))

    assert _update.get_physio_columns(tmp_path) == ["cardiac", "respiratory", "trigger"]


def test_get_physio_columns_first_participant_with_physio(tmp_path, monkeypatch):
    """The first participant with physio files, and a physio.json at the root."""
    import json

    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01", "sub-02"])
    _touch(tmp_path / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz")
    _touch(tmp_path / "sub-02" / "func" / "sub-02_task-rest_physio.tsv.gz")
    _touch(tmp_path / "physio.json", json.dumps({"Columns": ["cardiac"]}))

    assert _update.get_physio_columns(tmp_path) == ["cardiac"]


def test_get_nb_runs(tmp_path, monkeypatch):
    """The data files with a task entity of the first participant: not their companions."""
    from bids_inventory import _update

    monkeypatch.setattr(_update, "list_participants_in_dataset", lambda _: ["sub-01", "sub-02"])
    func = tmp_path / "sub-01" / "ses-1" / "func"
    for name in [
        "sub-01_ses-1_task-rest_run-1_bold.nii.gz",
        "sub-01_ses-1_task-rest_run-1_bold.json",
        "sub-01_ses-1_task-rest_run-1_sbref.nii.gz",
        "sub-01_ses-1_task-rest_run-1_physio.tsv.gz",
        "sub-01_ses-1_task-rest_run-2_bold.nii.gz",
        # multi-echo: one run
        "sub-01_ses-1_task-nback_echo-1_bold.nii.gz",
        "sub-01_ses-1_task-nback_echo-2_bold.nii.gz",
        "sub-01_ses-1_task-nback_events.tsv",
    ]:
        _touch(func / name)
    _touch(tmp_path / "sub-01" / "ses-1" / "eeg" / "sub-01_ses-1_task-rest_eeg.edf")
    # not the first participant
    _touch(tmp_path / "sub-02" / "func" / "sub-02_task-rest_bold.nii.gz")

    assert _update.get_nb_runs(tmp_path) == 4


def test_get_raw_size(tmp_path, monkeypatch):
    """The annexed files, without the derivatives and sourcedata folders."""
    import subprocess

    from bids_inventory import _update

    lines = [
        "100\tsub-01/anat/sub-01_T1w.nii.gz",
        "20\tsub-01/func/sub-01_task-rest_bold.nii.gz",
        "1000\tderivatives/fmriprep/sub-01/anat/sub-01_desc-preproc_T1w.nii.gz",
        "500\tsourcedata/sub-01.zip",
        # keys without size are not counted
        "unknown\tsub-01/func/sub-01_task-rest_events.tsv",
    ]
    listing = "\n".join(lines)
    monkeypatch.setattr(
        _update.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, listing, ""),
    )
    assert _update.get_raw_size(tmp_path) == 120

    monkeypatch.setattr(
        _update.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", "not a git repository"
        ),
    )
    assert _update.get_raw_size(tmp_path) is None
