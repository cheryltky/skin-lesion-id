# Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Download HAM10000 from Harvard Dataverse (non-commercial academic use per
its terms) and lay it out for training. The script is safe to re-run;
existing files are left in place:

```bash
python scripts/fetch_data.py --config config/smoke.yaml
```

Then train the smoke config (a few minutes on a laptop) or the full
dataset:

```bash
python scripts/train.py --config config/smoke.yaml
streamlit run app/streamlit_app.py
```

Run the tests:

```bash
pytest --cov=skinlesion
```

## Repository layout

```
skin-lesion-id/
├── assets/                  charts used by the README
├── config/
│   ├── default.yaml         all 7 diagnoses, full 10,015-image dataset
│   └── smoke.yaml           Rosendahl-only, frozen DINOv2, fast check
├── docs/                    setup and methodology
├── src/skinlesion/
│   ├── config.py            typed, validated, immutable configuration
│   ├── data/
│   │   ├── metadata.py      HAM10000_metadata.tab into typed records
│   │   ├── splits.py        lesion-grouped, leakage-free splitting
│   │   └── datasets.py      dataset, transforms, class weights
│   ├── modelling/
│   │   ├── model.py         backbone, device selection, checkpointing
│   │   ├── train.py         frozen warmup, early stopping
│   │   └── predict.py       calibrated inference with abstention
│   └── evaluation/
│       ├── metrics.py       top-k, macro-F1, malignant recall, confusions
│       └── calibration.py   ECE, reliability bins, temperature scaling
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_evaluate_model.ipynb
├── app/streamlit_app.py
├── scripts/
│   ├── fetch_data.py
│   └── train.py
└── tests/                   109 tests, 96% coverage
```
