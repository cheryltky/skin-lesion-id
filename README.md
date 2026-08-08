# Skin lesion diagnosis classification

Classifying dermatoscopic images of skin lesions, trained on
histopathology-confirmed cases from [HAM10000](https://doi.org/10.7910/DVN/DBW86T).

**This is a portfolio machine learning project, not a diagnostic device.**
It has not been clinically validated and must never substitute for
assessment by a qualified clinician. If you are concerned about a lesion
on your own skin, see a doctor -- nothing in this repository is medical
advice.

This repository contains the full pipeline: data acquisition from Harvard
Dataverse, a leakage-free dataset construction grouped by lesion, a
frozen-foundation-model classifier, an honest evaluation including
confidence calibration and a clinically-motivated recall metric, and a
Streamlit app.

---

## What this project is actually about

Training a classifier is not the interesting part. Four decisions matter
more, and each one is where a portfolio project usually goes wrong.

### 1. Images cluster within lesions, so a random split leaks

HAM10000 sometimes carries several dermatoscopic images of the same
lesion -- a follow-up shot, a slightly different crop, taken close
together in time. A naive image-level split scatters those
near-duplicates across train and test: the model partly memorises the
lesion and is then tested on almost the same photograph, and the
resulting accuracy is inflated and meaningless.

Every image from one lesion therefore lands in exactly one split. The
guarantee is enforced in [`splits.py`](src/skinlesion/data/splits.py) and
is checkable at runtime via `assert_no_leakage`, which the training script
calls before a single epoch runs -- the same guarantee, same mechanism, as
the leakage control in the companion
[nudibranch-id](https://github.com/cheryltky/nudibranch-id) project.

### 2. The class distribution is long tailed, and misclassifications are not equally bad

Melanocytic nevi (ordinary moles) vastly outnumber every other diagnosis,
because most of what a skin cancer clinic biopsies turns out benign.
Plain accuracy rewards a model that learns only to call everything a
nevus, so evaluation reports **macro-F1** alongside accuracy, and training
weights the loss by inverse class frequency.

That is not the whole story, though. Confusing two benign diagnoses is a
false alarm; calling a melanoma benign is a missed cancer. Evaluation
therefore also reports **malignant recall**: of the lesions that are
actually malignant or pre-malignant, what fraction did the model flag as
such, regardless of which specific malignant subtype it guessed. See
[`metrics.py`](src/skinlesion/evaluation/metrics.py).

### 3. With this little data, a frozen foundation model beats fine-tuning

`config/smoke.yaml` trains on roughly 1,550 lesions -- not enough to
fine-tune a network from scratch without either overfitting or wasting
most of its capacity relearning generic visual features a much larger
dataset already taught some other model. The backbone is a frozen
[DINOv2](https://arxiv.org/abs/2304.07193)
(`vit_small_patch14_dinov2.lvd142m`) with only a linear head trained on
top -- `warmup_frozen_epochs` equals `epochs`, so the backbone is never
unfrozen. This is the same frugal-AI choice validated on the
[nudibranch-id](https://github.com/cheryltky/nudibranch-id) project, where
it took a five-species smoke split from 0.774 to 1.000 top-1 accuracy.

That result does not repeat here, and the honest reason why is itself
worth stating: melanoma diagnosis from a single image is a genuinely hard
problem, one trained dermatologists get wrong at meaningful rates, and
this recipe reaches 0.562 top-1 accuracy on this task at this scale. See
Results below and [decision 4](#4-the-abstain-threshold-does-not-do-what-you-might-assume).

### 4. The abstain threshold does not do what you might assume

The app declines to name a diagnosis below a calibrated confidence
threshold, the same honesty mechanism as the nudibranch project. There,
raising the threshold reliably traded coverage for accuracy. Here, raising
it from 0.00 to 0.60 does take overall accuracy from 0.562 to 0.806 --
but **malignant recall barely moves, from 0.590 to 0.583** over that same
range. The model's confidence and its correctness on the malignant/benign
question are not well aligned at moderate thresholds. Malignant recall
only meaningfully recovers at thresholds so high that image coverage
collapses to 20% and then 5%. As configured, the abstain mechanism mostly
governs how often the model is willing to commit to *a* diagnosis; it
does not, on this evidence, reliably protect against the specific mistake
that matters most. See the Results table below and notebook 3's summary
for the full picture, including which threshold would actually be needed.

---

## Notebooks

Outputs are committed, so these read on GitHub as a finished analysis --
tables and charts already rendered, nothing to run:

| Notebook | What it covers |
|---|---|
| [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) | The images-per-lesion leakage problem, class balance, ground-truth confidence tiers, who the data represents, what the images look like |
| [`02_train_model.ipynb`](notebooks/02_train_model.ipynb) | Verifying the split is clean, class weights, training curves, fitting and saving the calibration temperature |
| [`03_evaluate_model.ipynb`](notebooks/03_evaluate_model.ipynb) | Accuracy and macro-F1, malignant recall, which diagnoses are confused with what, calibration, the abstain-threshold trade-off, and an honest summary of what the results do and don't show |

All three are executed against [`config/smoke.yaml`](config/smoke.yaml),
restricted to the Rosendahl subset described below, so the published
numbers are reproducible from a laptop in a few minutes once the data is
downloaded.

## Why the Rosendahl (Queensland) subset

`config/smoke.yaml` restricts training to 2,259 images from **Cliff
Rosendahl's skin cancer practice at the University of Queensland**,
ethics-approved (Protocol No. 2017001223) and **100% confirmed by
histopathology** -- a biopsy result, not a clinical impression.

Two honest reasons for this choice, not one. It is the identifiably
Australian slice of HAM10000. But independent of geography, it is also the
*highest-confidence ground truth in the dataset*: across HAM10000 as a
whole, only about 53% of labels are histopathology-confirmed, and the rest
rely on follow-up examination, clinical consensus among dermatologists, or
confocal microscopy -- reasonable methods, but not as certain as a biopsy.
Training on the most reliable labels available is defensible on its own
merits.

`config/default.yaml` uses the full 10,015-image dataset across all seven
diagnoses, and is *not* restricted to a frozen backbone -- see the config
file for the reasoning. It has not been run as part of this repository;
the smoke config's frugal-AI result is what's actually validated here.

---

## Results

From the smoke run in [`config/smoke.yaml`](config/smoke.yaml) -- 5
diagnoses (`akiec`, `bcc`, `bkl`, `mel`, `nv`; `df` and `vasc` were dropped
for having too few images in this subset), 1,557 training images, frozen
DINOv2 backbone, linear head trained for 10 epochs:

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.562 |
| Top-3 accuracy | 0.916 |
| Macro-F1 | 0.486 |
| **Malignant recall** | **0.590** |
| Calibration error before scaling | 0.093 |
| Calibration error after scaling | **0.052** |
| Fitted temperature | 1.290 |

Chance level across 5 classes is 0.200, so this is well above chance but a
genuinely moderate result -- not a claim of clinical usefulness. Melanoma
is the weakest class (F1 0.265), and the single most frequent mistake in
the whole test set is exactly the dangerous one: **melanoma called an
ordinary mole, 25 times out of 333 test images.**

The abstain threshold trades coverage for accuracy, but *not* proportionally
for malignant recall:

| Threshold | Images answered | Accuracy | Malignant recall |
|---|---|---|---|
| 0.00 | 100.0% | 0.562 | 0.590 |
| 0.50 (configured) | 54.1% | 0.711 | 0.585 |
| 0.60 | 38.7% | 0.806 | 0.583 |
| 0.75 | 20.4% | 0.926 | 0.800 |
| 0.90 | 5.1% | 1.000 | 1.000 |

Read that table as a limitation, not a success story: the threshold that
would actually protect malignant recall costs the vast majority of
coverage. See [notebook 3's summary](notebooks/03_evaluate_model.ipynb)
for the full reasoning, including why this differs so much from the
near-perfect result the same recipe produced on the nudibranch project
(support correlates far more strongly with F1 here, r = 0.80 versus 0.52,
consistent with a genuinely harder visual problem rather than a bug).

---

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Download HAM10000 from Harvard Dataverse (non-commercial academic use per
its terms) and lay it out for training -- safe to re-run, existing files
are left in place:

```bash
python scripts/fetch_data.py --config config/smoke.yaml
```

Then train the smoke config (a few minutes on a laptop) or the full
dataset:

```bash
python scripts/train.py --config config/smoke.yaml
streamlit run app/streamlit_app.py
```

---

## Repository layout

```
skin-lesion-id/
├── config/
│   ├── default.yaml         all 7 diagnoses, full 10,015-image dataset
│   └── smoke.yaml           Rosendahl-only, frozen DINOv2, fast check
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

---

## Tests

```bash
pytest --cov=skinlesion
```

109 tests, 96% statement coverage. The tests worth reading are
[`test_splits.py`](tests/test_splits.py), which proves the leakage
guarantee directly, and
[`test_metrics.py`](tests/test_metrics.py), which checks that
`malignant_recall` counts a missed melanoma as a miss regardless of which
benign class the model guessed instead -- the property the whole metric
exists to guarantee.

---

## Known limitations

- **Closed-set.** No way to say "this is not one of the diagnoses I know."
  Anything else is forced into the nearest known class.
- **Dermatoscope-only.** Trained and evaluated on dermatoscopic images,
  not ordinary phone photos, which look meaningfully different.
- **Demographic scope.** HAM10000 is not documented by Fitzpatrick skin
  type, and dermoscopy archives are known to skew toward lighter skin
  tones. Performance on darker skin is untested and should not be assumed
  to match the figures above.
- **The abstain threshold's limits are the headline finding, not a
  footnote.** See Results above.
- **Not validated at the full-dataset scale.** Whether a frozen head still
  wins over fine-tuning with all 10,015 images and seven diagnoses is
  untested; `config/default.yaml` deliberately does not copy the smoke
  config's architecture choice without evidence for the harder case.

---

## Data and licensing

HAM10000 is downloaded fresh from Harvard Dataverse
(doi:10.7910/DVN/DBW86T) rather than redistributed here, under its
non-commercial academic use terms. Code in this repository is MIT
licensed; the dataset is not mine to license, and this project is not a
medical device.
