# Reproducing this package

Requires Python with the packages in `code/requirements.txt` (numpy, pandas, scipy, matplotlib, networkx). No NBA requests or API credentials are needed for these calculations: the derived count matrices and frozen training inputs are included.

From this `final` folder:

    python -m pip install -r code/requirements.txt
    python code/reproduce_final.py --verify-only
    python code/reproduce_final.py

The first command installs dependencies if needed. Verify-only reconstructs the minute and initial separate-role rules across all180 cases. The complete run refits the previous-season-only model and observed entropy regressions, saving to `reproduced_outputs/` by default. Existing published outputs are preserved. Use `--destination` to choose another result folder.

The prediction model itself can be run with:

    python code/previous_season_history.py --output-dir data/nba_passing_output --destination reproduced_outputs/prediction_model

The regression wrapper can be run with:

    python code/run_regressions.py --package-root . --destination reproduced_outputs/regressions

Historical training retains the original15-early-present/10-later-absent historical eligibility. Final evaluation requires10+ later absences and a nonempty observed network. The regression wrapper creates a temporary eligibility copy without changing historical-training metadata.

Source data here are calculation inputs derived from saved NBA tracking profiles and logs. This package does not reproduce the original network downloads or independently establish every source record's completeness. Frozen JSON source paths may identify their original computer locations; the reproduction scripts use packaged relative inputs and generate new provenance paths.
