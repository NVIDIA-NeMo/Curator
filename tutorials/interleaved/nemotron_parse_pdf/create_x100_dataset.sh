#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:?usage: create_x100_dataset.sh SOURCE DESTINATION}
DESTINATION=${2:?usage: create_x100_dataset.sh SOURCE DESTINATION}

if [[ -e "${DESTINATION}" ]]; then
  echo "Destination already exists: ${DESTINATION}" >&2
  exit 1
fi

mkdir -p "${DESTINATION}"
export SOURCE DESTINATION

# Force independent physical copies. Do not allow CoW reflinks or hard links.
seq -w 0 99 | xargs -P 8 -I '{}' bash -c '
  replica="copy-${1}"
  mkdir -p "${DESTINATION}/${replica}"
  cp -a --reflink=never "${SOURCE}/." "${DESTINATION}/${replica}/"
  echo "${replica} complete"
' _ '{}'

for replica in $(seq -w 0 99); do
  sed "s|\"file_name\": \"|\"file_name\": \"copy-${replica}/pdfs/|" \
    "${SOURCE}/manifest.jsonl"
done >"${DESTINATION}/manifest.jsonl"

source_pdfs=$(find "${SOURCE}/pdfs" -type f -iname '*.pdf' | wc -l)
copied_pdfs=$(find "${DESTINATION}" -type f -iname '*.pdf' | wc -l)
manifest_rows=$(wc -l <"${DESTINATION}/manifest.jsonl")
expected=$((source_pdfs * 100))

if [[ "${copied_pdfs}" -ne "${expected}" || "${manifest_rows}" -ne "${expected}" ]]; then
  echo "Validation failed: expected=${expected} pdfs=${copied_pdfs} manifest=${manifest_rows}" >&2
  exit 1
fi

echo "Validated ${copied_pdfs} PDFs and ${manifest_rows} manifest rows"
