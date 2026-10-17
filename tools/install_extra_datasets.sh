#! /bin/bash

# Installs the BIDS datasets that are not on openneuro:
# - from the datalad superdataset: abide, abide2, adhd200, corr
# - from courtois-neuromod: cneuromod
# - other BIDS datasets of the datalad superdataset, in the indi, labs and studyforrest folders
#
# Usage:
#   install_extra_datasets.sh [target_dir]   (default: ./tmp)
#
# Then set 'local_paths.non_openneuro' to target_dir in your configuration
# and run: bids_inventory update --source non-openneuro

set -eux

target_dir=$(realpath "${1:-./tmp}")
mkdir -p "${target_dir}"

datalad install -s ///abide/RawDataBIDS/ "${target_dir}/abide"
datalad install -s ///abide2/RawData/ "${target_dir}/abide2"
datalad install -s ///adhd200/RawDataBIDS/ "${target_dir}/adhd200"
datalad install -s ///corr/RawDataBIDS/ "${target_dir}/corr"

for sds in abide abide2 adhd200 corr; do
    (cd "${target_dir}/${sds}" && datalad -f '{path}' subdatasets | xargs -n 1 -P 10 datalad install)
done

for sds in abide abide2 adhd200 corr; do
    for ds in "${target_dir}/${sds}"/*/; do
        (cd "${ds}" && datalad get ./*.tsv) || echo 'no tsv'
        (cd "${ds}" && datalad get ./*.json) || echo 'no json'
    done
done

# TODO get neuromod derivatives ?
datalad install -s https://github.com/courtois-neuromod/cneuromod.git "${target_dir}/cneuromod"
(cd "${target_dir}/cneuromod" && datalad -f '{path}' subdatasets | xargs -n 1 -P 2 datalad install)

# other BIDS datasets of the datalad superdataset (found with tools/query_datalad_registry.py),
# grouped by collection: <collection>/<dataset> <source in the datalad superdataset>
datasets="
indi/HypnosisBarrios ///indi/HypnosisBarrios/RawData
indi/PRIME_mountsinai-P ///indi/PRIME/mountsinai-P
indi/PRIME_mountsinai-S ///indi/PRIME/mountsinai-S
indi/PRIME_newcastle ///indi/PRIME/newcastle
indi/PRIME_nki ///indi/PRIME/nki
indi/PRIME_princeton ///indi/PRIME/princeton
indi/PRIME_rockefeller ///indi/PRIME/rockefeller
indi/PRIME_ucdavis ///indi/PRIME/ucdavis
indi/PRIME_uminn ///indi/PRIME/uminn
labs/gobbini_famface ///labs/gobbini/famface/data
labs/haxby_life ///labs/haxby/life
labs/haxby_raiders ///labs/haxby/raiders
studyforrest/3T_multiresolution_fmri ///studyforrest/original/3T_multiresolution_fmri
studyforrest/7T_multiresolution_fmri ///studyforrest/original/7T_multiresolution_fmri
studyforrest/phase2 ///studyforrest/original/phase2
"
echo "${datasets}" | while read -r ds source; do
    if [ -z "${ds}" ]; then
        continue
    fi
    mkdir -p "$(dirname "${target_dir}/${ds}")"
    datalad install -s "${source}" "${target_dir}/${ds}"
    (cd "${target_dir}/${ds}" && datalad get ./*.tsv) || echo 'no tsv'
    (cd "${target_dir}/${ds}" && datalad get ./*.json) || echo 'no json'
done
