#!/bin/bash
# Download CCPD for the live-predict path only. The demo works without it,
# from data/precomputed_*.csv. CCPD2020.zip is ~908 MB; CCPD2019 is ~13 GB,
# so the trial only used a Range-downloaded partial copy for the blue control.
# Sources (as recorded in TRIAL_REPORT.md):
#   Zenodo record 15647076 (CCPD archive, original repo detectRecog/CCPD)
#   CCPD2020.zip MD5 eb93e88c5988879f8da6f92bbf083324
set -e
mkdir -p data
echo "Download CCPD2020.zip into data/ manually from the Zenodo record above,"
echo "then unzip it so that data/CCPD2020/ccpd_green/test/*.jpg exists."
echo "For a blue control, the trial used only the ccpd_fn subset of CCPD2019."
echo "This script intentionally does NOT auto-download 13 GB."
