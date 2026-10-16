#! /bin/bash

# Builds the dashboard (bids_inventory/browse_app.py) as a static website
# that runs in the browser (WebAssembly), for example on GitHub Pages.
#
# Usage, from the root of the repository:
#   tools/build_dashboard.sh [output_dir]   (default: _site)
#
# Needs: pip install build marimo
# To look at it locally: python -m http.server --directory _site

set -eu

output_dir=$(realpath -m "${1:-_site}")
build_dir=$(mktemp -d)
trap 'rm -rf "${build_dir}"' EXIT

# bids_inventory is not on PyPI: the notebook installs this wheel in the browser
mkdir -p "${build_dir}/public"
python -m build --wheel --outdir "${build_dir}/public" .
basename "$(ls "${build_dir}"/public/*.whl)" > "${build_dir}/public/wheel.txt"

# the files in public/ next to the notebook are copied to the website
cp bids_inventory/browse_app.py "${build_dir}/"
marimo export html-wasm "${build_dir}/browse_app.py" \
    --output "${output_dir}" \
    --mode run \
    --no-show-code \
    --force

echo "Dashboard built in ${output_dir}"
