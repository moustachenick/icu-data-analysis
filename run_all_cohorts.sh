#!/bin/bash
# Run one mode for the full cohort and then for the TBI subgroup.
# Usage: ./run_all_cohorts.sh [classification|regression]   (default: classification)
# The cached CSVs do not include the cohort in their names, so they are removed before each
# run and at the end. Outputs: output/{mode}_<timestamp>.txt and output/{mode}_TBI_<timestamp>.txt
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-classification}"

for COHORT in "" "TBI"; do
  ./clean_generated_files.sh
  python src/main.py --mode "$MODE" --pathology "$COHORT"
done
./clean_generated_files.sh
