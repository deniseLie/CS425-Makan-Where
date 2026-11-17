# Singlish-to-English Normalizer

## Overview

The normalizer converts Singapore Colloquial English (Singlish) into
standard English while preserving the original meaning and user intent.

The model is based on Google's FLAN-T5-base and is fine-tuned for
Singlish-to-English text normalization.

Hugging Face model (served by the backend):
makan-where-cs425/makan-flan-normalizer-masked

It replaces makan-where-cs425/makan-flan-normalizer, the cleaned-data model below. The masked model
is trained on the same cleaned SEAT split plus 1,000 synthetic restaurant requests, with places,
dishes, cuisines and dietary words replaced by typed placeholders (`in tampines got halal food` ->
`in Place1 got Diet1 food`), which is how the pipeline sends them at run time
(backend/src/normalizer/entities.py). Data: `python training/build_normalizer_data.py`; training:
`training/normalizer_masked_kaggle.ipynb`.

| Model (same pipeline, typed placeholders) | SEAT test BLEU / chrF | Gold BLEU / chrF | Gold returned unchanged |
|---|---:|---:|---:|
| makan-flan-normalizer | 54.5 / 68.9 | 12.5 / 44.7 | 46% |
| makan-flan-normalizer-masked | 53.8 / 68.1 | 22.0 / 54.9 | 0% |

(`python eval/eval_normalizer.py`, 17 Nov 2026.)

## Dataset

The dataset contains paired text examples with two columns:

- `singlish`: Original Singlish input.
- `english`: Corresponding English normalization target.

The data was divided into:

- 80% training
- 10% validation
- 10% testing

The split used `random_state=42`.

The test split was kept separate from model training.

## Data Cleaning

A cleaned training experiment was conducted using the original
training split.

The cleaning process included:
- Manually correcting seven training examples.
- Removing two examples whose intended meaning was too uncertain
  to repair reliably.
- Applying corrections only to the training split.

The cleaned training data was saved as `train_cleaned_pilot.csv`.

## Model Training

Base model: `google/flan-t5-base`

Cleaned-data experiment configuration:

- Epochs: 3
- Learning rate: 3e-5
- Training batch size: 4
- Evaluation batch size: 4
- Evaluation strategy: Each epoch
- Model selection: Best validation loss
- Random seed: 42

The notebook also contains the original training experiment and
comparison between original and cleaned checkpoints.

See `normalizer_training.ipynb` for the implementation.

## Evaluation

### FLAN-T5 prediction comparison

`model_comparison.csv` contains 161 examples comparing:
- Base FLAN-T5 predictions
- Fine-tuned FLAN-T5 predictions
- Reference English translations

### Downstream normalization evaluation

The following manually reviewed evaluations assess meaning preservation.
They compare normalization approaches in the wider pipeline and should
not be interpreted as direct FLAN-T5 benchmark scores.

| Evaluation | Examples | Baseline / Rules-only | Hybrid |
|---|---:|---:|---:|
| Final test comparison | 20 | 20/20 (100%) | 19/20 (95%) |
| Unseen evaluation | 20 | 17/20 (85%) | 19/20 (95%) |
| Diagnostic evaluation | 31 | 25/31 (80.6%) | 31/31 (100%) |

The evaluations indicate that the hybrid approach better preserved
meaning on the unseen and diagnostic examples. The final test
comparison also shows one hybrid failure, so the results should be
interpreted alongside the individual examples.

## Files

- `normalizer_training.ipynb`: Training, cleaning and checkpoint
  comparison experiments.
- `model_comparison.csv`: Base versus fine-tuned FLAN-T5 predictions.
- `normalizer_unseen_eval_reviewed.csv`: Reviewed unseen examples.
- `normalizer_diagnostic_reviewed.csv`: Reviewed diagnostic examples.
- `normalizer_final_test_predictions.csv`: Rules-only versus
  hybrid-guarded pipeline comparison.

## Reproducibility

The notebook contains the data preparation, training and evaluation
experiments.

The dataset CSVs and model checkpoints are maintained separately
from this repository. Refer to the Hugging Face model repository
for the deployed normalizer weights.

Dataset redistribution permissions should be checked before
publishing the original or split CSV files.
