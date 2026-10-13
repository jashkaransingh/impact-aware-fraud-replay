# Impact-aware fraud replay

Research code for bounded replay in sequential fraud classification. The replay experiments compare no replay, random replay, forgetting-based selection, amount-based selection, and a combined score. Transaction amount is an evaluation and selection proxy, not a measured financial loss.

## Code and attribution

- `upstream/POCL/` preserves the [POCL](https://github.com/finint/POCL) implementation of Zhang et al., *Pre-trained Online Contrastive Learning for Insurance Fraud Detection* (AAAI 2024), at commit `ef50bc19fd82c18374be326c073b41ac69f217ae`. Its original README and GPL-3.0 license remain there. The raw upstream CSVs are excluded from this repository.
- `upstream/handbook/` preserves the [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) simulator source and notebook at commit `81cf7d1714bb7b2f5b496407d9055d91dc68dc25`, with its license and provenance. `simulator.py` extracts the notebook's function definitions and adds imports.
- `src/impact_replay/`, `scripts/`, `configs/`, and `tests/` are this project's replay pipeline, experiments, and checks. The tabular replay model is separate from POCL's graph model. `scripts/prepare_audited_pocl.py` generates an explicitly corrected POCL variant in ignored `work/`; it does not change the preserved upstream files. See `scripts/audited-corrections.patch`.

The root `LICENSE` and the licenses inside both `upstream/` directories contain the applicable GPL terms. Please retain the upstream credits and cite the original works when using their code or data generator.

## Data

Data is not distributed here. For POCL, obtain `claims.csv` and `bene.csv` from the [upstream POCL dataset](https://github.com/finint/POCL/tree/main/dataset) and place them in `upstream/POCL/dataset/`. The Handbook data are generated locally by `scripts/make_handbook.py` from the included simulator. For the IEEE-CIS development experiment, obtain `train_transaction.csv` through the [Kaggle competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) under its access terms and place it in `data/ieee-cis/`. Do not publish the downloaded CSV or generated feature matrices.

## Setup and runs

Python 3.12 was used for the original project runs on Linux CPU. A clean macOS rebuild has not been checked. Install the dependencies in an environment you create locally; `requirements.lock` records the versions used for the earlier runs. PyTorch CPU wheels may require the [PyTorch wheel index](https://download.pytorch.org/whl/cpu).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export PYTHONPATH=src
python -m unittest discover -s tests -v
```

The commands below write to ignored local directories. Choose unused output paths; the runners reject existing ones.

```bash
# Handbook simulation and one configurable replay suite
python scripts/make_handbook.py
python scripts/run_suite.py --config configs/operating-point.json --prefix new-suite

# Controlled A→B→A mechanism test; the protocol and full-suite scripts are included
python -m impact_replay.controlled --config configs/controlled-validation.json --out controlled/new-validation --seed 101 --policy none
python scripts/run_controlled_test.py
python scripts/verify_controlled.py --prefix test

# IEEE-CIS development-only preparation and fixed baseline comparison
python scripts/prepare_ieee_cis.py
python scripts/prepare_ieee_windows.py
python scripts/run_ieee_no_replay_baseline.py
python scripts/run_ieee_balanced_replay.py
```

To run preserved POCL, first preprocess its raw CSVs using the upstream code's expected `Dataset` path, then run the wrapper from the project root:

```bash
mkdir -p work/pocl
cp upstream/POCL/preprocess.py work/pocl/
ln -s ../../upstream/POCL/dataset work/pocl/Dataset
(cd work/pocl && OPENBLAS_NUM_THREADS=1 python preprocess.py)
python scripts/run_pocl.py --pretrain-epochs 300 --offline-epochs 100 --online-steps 10 --days 364 --out results/new-pocl-full
```

`scripts/verify_results.py` verifies the historical `results/final-*` runs when their local artifacts are present. `scripts/verify_controlled.py` similarly needs the full local controlled run artifacts; the compact published summaries alone do not satisfy those checks.

## Existing results

The small files in `summaries/` were calculated from existing local outputs after their recorded checks passed. `controlled.csv` gives five-seed means for both controlled budgets and all three amount assignments. With independent amounts and the 256-slot / 128-retrieval budget, A average precision after B was 0.050 with no replay, 0.888 with random replay, 0.937 with forgetting selection, and 0.931 with the combined score. This synthetic setting demonstrates forgetting and retention, not real-world fraud savings.

The released POCL code's full schedule produced mean daily AUC 0.816527, F1 0.647858, and accuracy 0.772836 across 350 online days; a same-seed rerun matched daily metrics. This is a run of released code, not an exact numerical replication of the paper. The separately corrected variant changed several behaviors together and is not an ablation.

On IEEE-CIS development data, one fixed-seed class-balanced random-replay comparison had mean final AP 0.192821 over five historical held-out windows, versus 0.189670 without replay. The sixth window AP was 0.184765 versus 0.185013. This is preliminary and does not test impact-aware selection. The chronological validation and reserved evaluation partitions were not used in this comparison.

## Limits

The controlled contexts are synthetic. The Handbook simulator has amount-dependent fraud rules. IEEE-CIS uses a limited initial feature set and one development seed; validation and reserved evaluation results are absent. Amount-weighted recall uses transaction amount as a proxy and cannot establish recovered money. Replay budgets, labels, and immediate feedback assumptions do not model an operational fraud system. No training-resume command or clean-environment rebuild has been verified.
