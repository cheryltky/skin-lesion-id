# Methodology

Design decisions, validation and full results for the smoke run. The
README carries the summary and the recommendations.

## Decision 1: images cluster within lesions, so a random split leaks

HAM10000 sometimes carries several dermatoscopic images of the same
lesion: a follow-up shot, a slightly different crop, taken close together
in time. A naive image-level split scatters those near-duplicates across
train and test. The model partly memorises the lesion and is then tested
on almost the same photograph, so the resulting accuracy is inflated and
meaningless.

Every image from one lesion therefore lands in exactly one split. The
guarantee is enforced in [`splits.py`](../src/skinlesion/data/splits.py)
and is checkable at runtime via `assert_no_leakage`, which the training
script calls before a single epoch runs. It is the same guarantee and
mechanism as the leakage control in the companion
[nudibranch-id](https://github.com/cheryltky/nudibranch-id) project.

## Decision 2: a long-tailed class distribution and unequal error costs

Melanocytic nevi (ordinary moles) vastly outnumber every other diagnosis,
because most of what a skin cancer clinic biopsies turns out benign.
Plain accuracy rewards a model that learns only to call everything a
nevus, so evaluation reports **macro-F1** alongside accuracy, and training
weights the loss by inverse class frequency.

Confusing two benign diagnoses is a false alarm; calling a melanoma benign
is a missed cancer. Evaluation therefore also reports **malignant
recall**: of the lesions that are actually malignant or pre-malignant, what
fraction did the model flag as such, regardless of which specific
malignant subtype it guessed. See
[`metrics.py`](../src/skinlesion/evaluation/metrics.py).

## Decision 3: a frozen foundation model instead of fine-tuning

`config/smoke.yaml` trains on 1,557 images from 1,093 lesions (earlier
text said roughly 1,550 lesions). That is not enough to fine-tune a network from scratch without either overfitting or wasting
most of its capacity relearning generic visual features that a much larger
dataset already taught some other model. The backbone is a frozen
[DINOv2](https://arxiv.org/abs/2304.07193)
(`vit_small_patch14_dinov2.lvd142m`) with only a linear head trained on
top. `warmup_frozen_epochs` equals `epochs`, so the backbone is never
unfrozen. This is the same frugal-AI choice validated on the
[nudibranch-id](https://github.com/cheryltky/nudibranch-id) project, where
it took a five-species smoke split from 0.774 to 1.000 top-1 accuracy.

That result does not repeat here. Melanoma diagnosis from a single image
is a genuinely hard problem, one trained dermatologists get wrong at
meaningful rates, and this recipe reaches 0.562 top-1 accuracy on this
task at this scale.

## Decision 4: the abstain threshold does not do what you might assume

The app declines to name a diagnosis below a calibrated confidence
threshold, the same honesty mechanism as the nudibranch project. There,
raising the threshold reliably traded coverage for accuracy. Here, raising
it from 0.00 to 0.60 does take overall accuracy from 0.562 to 0.806, but
**malignant recall barely moves, from 0.590 to 0.583** over that same
range. The model's confidence and its correctness on the malignant/benign
question are not well aligned at moderate thresholds. Malignant recall
only meaningfully recovers at thresholds so high that image coverage
collapses to 20% and then 5%. As configured, the abstain mechanism mostly
governs how often the model is willing to commit to a diagnosis; it does
not, on this evidence, reliably protect against the specific mistake that
matters most. A production version would need a threshold, or a separate
decision rule, tuned against malignant recall rather than overall
accuracy.

## Why the Rosendahl (Queensland) subset

`config/smoke.yaml` restricts training to 2,259 images from Cliff
Rosendahl's skin cancer practice at the University of Queensland,
ethics-approved (Protocol No. 2017001223) and 100% confirmed by
histopathology, which is a biopsy result rather than a clinical
impression.

There are two reasons. It is the identifiably Australian slice of
HAM10000. Independent of geography, it is also the highest-confidence
ground truth in the dataset: across HAM10000 as a whole, only about 53% of
labels are histopathology-confirmed, and the rest rely on follow-up
examination, clinical consensus among dermatologists, or confocal
microscopy. These are reasonable methods but not as certain as a biopsy.
Training on the most reliable labels available is defensible on its own
merits.

`config/default.yaml` uses the full 10,015-image dataset across all seven
diagnoses and is not restricted to a frozen backbone; see the config file
for the reasoning. It has not been run as part of this repository. The
smoke config's frugal-AI result is what is validated here.

After the minimum-images filter, the smoke run keeps 2,226 images from
1,534 distinct lesions across 5 diagnoses. The `df` (30 images) and `vasc`
(3 images) classes fall below the 50-image minimum and are dropped. The
largest to smallest class ratio is 2.7x. The split is 1,557 train, 336
validation and 333 test images (1,093, 218 and 223 lesions), with zero
lesion overlap between splits. Class weights (inverse frequency,
normalised to mean 1) run from 0.48 for `nv` to 1.31 for `akiec`.

## Notebooks

Outputs are committed, so these read on GitHub as a finished analysis,
with tables and charts already rendered:

| Notebook | What it covers |
|---|---|
| [`01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb) | The images-per-lesion leakage problem, class balance, ground-truth confidence tiers, who the data represents, what the images look like |
| [`02_train_model.ipynb`](../notebooks/02_train_model.ipynb) | Verifying the split is clean, class weights, training curves, fitting and saving the calibration temperature |
| [`03_evaluate_model.ipynb`](../notebooks/03_evaluate_model.ipynb) | Accuracy and macro-F1, malignant recall, which diagnoses are confused with what, calibration, the abstain-threshold trade-off, and a summary of what the results do and do not show |

All three are executed against [`config/smoke.yaml`](../config/smoke.yaml),
restricted to the Rosendahl subset, so the published numbers are
reproducible from a laptop in a few minutes once the data is downloaded.

## Results

From the smoke run: 5 diagnoses (`akiec`, `bcc`, `bkl`, `mel`, `nv`),
1,557 training images, frozen DINOv2 backbone, linear head trained for 10
epochs (early stopping ended the published run at epoch 5, with best
validation accuracy 0.5863 at epoch 1).

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.562 |
| Top-3 accuracy | 0.916 |
| Macro-F1 | 0.486 |
| **Malignant recall** | **0.590** |
| Calibration error before scaling | 0.093 |
| Calibration error after scaling | **0.052** |
| Fitted temperature | 1.290 |

Chance level across 5 classes is 0.200. Of 139 malignant or pre-malignant
test lesions, 82 were caught and 57 missed.

Per-diagnosis performance on the test split:

| Diagnosis | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `mel` Melanoma | 51 | 0.277 | 0.255 | 0.265 |
| `akiec` Actinic keratosis / intraepithelial carcinoma | 44 | 0.409 | 0.409 | 0.409 |
| `bkl` Benign keratosis | 73 | 0.483 | 0.397 | 0.436 |
| `bcc` Basal cell carcinoma | 44 | 0.645 | 0.455 | 0.533 |
| `nv` Melanocytic nevus | 121 | 0.709 | 0.884 | 0.787 |

Most frequent confusions (true class to predicted class):

| Count | True | Predicted | Malignant to benign |
|---|---|---|---|
| 25 | `mel` | `nv` | yes |
| 20 | `bkl` | `mel` | no |
| 15 | `akiec` | `bkl` | yes |
| 11 | `bcc` | `akiec` | no |
| 11 | `bkl` | `akiec` | no |
| 10 | `nv` | `bkl` | no |
| 8 | `bcc` | `nv` | yes |
| 8 | `bkl` | `nv` | no |
| 6 | `mel` | `bkl` | yes |
| 5 | `akiec` | `mel` | no |

The abstain threshold trades coverage for accuracy, but not
proportionally for malignant recall:

| Threshold | Images answered | Accuracy | Malignant recall |
|---|---|---|---|
| 0.00 | 100.0% | 0.562 | 0.590 |
| 0.30 | 96.7% | 0.568 | 0.586 |
| 0.50 (configured) | 54.1% | 0.711 | 0.585 |
| 0.60 | 38.7% | 0.806 | 0.583 |
| 0.75 | 20.4% | 0.926 | 0.800 |
| 0.90 | 5.1% | 1.000 | 1.000 |

The threshold that would actually protect malignant recall costs the
vast majority of coverage. Support correlates more strongly with F1 here
than on the nudibranch project (r = 0.80 versus 0.52), consistent with a
genuinely harder visual problem rather than a bug. Melanoma and basal
cell carcinoma are diagnoses that trained dermatologists get wrong at
meaningful rates from an image alone, and 2,259 training images is little
data for five classes with this much visual overlap. This is the limit of
the frugal-AI approach for this task at this scale, not evidence it was
the wrong thing to try.

Calibration: the model was overconfident before scaling, the usual
direction, unlike the underconfident pattern on the nudibranch smoke
runs. The correction is real, but calibration alone does not address the
malignant-recall problem; a well-calibrated model can still be
systematically wrong about one important class.

## Tests

```bash
pytest --cov=skinlesion
```

109 tests, 96% statement coverage. The tests worth reading are
[`test_splits.py`](../tests/test_splits.py), which proves the leakage
guarantee directly, and [`test_metrics.py`](../tests/test_metrics.py),
which checks that `malignant_recall` counts a missed melanoma as a miss
regardless of which benign class the model guessed instead.

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
  footnote.**
- **Not validated at the full-dataset scale.** Whether a frozen head still
  wins over fine-tuning with all 10,015 images and seven diagnoses is
  untested; `config/default.yaml` deliberately does not copy the smoke
  config's architecture choice without evidence for the harder case.
