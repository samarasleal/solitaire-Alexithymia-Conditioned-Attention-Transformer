# Trait-Conditioned Longitudinal Attention for Speech-Based Prediction of Depression Outcomes

## Overview

This repository contains the Python code developed for longitudinal speech-based prediction of depression outcomes in young patients undergoing internet-based cognitive behavioral therapy (iCBT). The proposed Conditioned Longitudinal Attention (CoLA) framework uses a patient-specific trait representation to guide attention across treatment sessions.

The principal model, **CoLA-Trait**, directly uses standardized baseline alexithymia as the conditioning signal. An additional **CoLA-Speech** variant derives the conditioning representation from speech recorded during the baseline session, while baseline alexithymia is used only as an auxiliary supervision target during training.

The repository includes code for data preparation, target construction, fold-specific speech encoding, longitudinal modeling, baseline experiments, ablation studies, and statistical evaluation. Clinical data and raw audio are not included because of privacy and ethical restrictions.

## Repository structure

```text
Data_Preprocessing/
  __init__.py
  DataPreprocessing_melt.ipynb
  DataPreprocessing_melt.py

Data_Processing/
  __init__.py
  Run_CNN.ipynb
  Run_CNN.py
  Run_RNN.ipynb
  Run_RNN.py
  TargetVariables.ipynb
  TargetVariables.py

Paper_TaCoLA/
  BASE_BIN.ipynb
  BASE_REG.ipynb
  CNN_Extractor.ipynb
  COLA_BIN.ipynb
  COLA_REG.ipynb
  Main_Metrics.ipynb
  Seeds_Metrics.ipynb

config.yaml
LICENSE
README.md
```

## Method overview

The analysis comprises three main stages:

1. **Fold-specific speech encoding:** Speech segments are converted into log-Mel spectrograms. Within each leave-one-patient-out (LOPO) fold, a CNN encoder is trained exclusively on the training patients and used to extract segment-level embeddings for the corresponding training and held-out partitions.
2. **Trait-conditioned longitudinal modeling:** Segment embeddings are aggregated into session representations and processed by a temporal Transformer. CoLA uses either standardized baseline alexithymia or a baseline-speech-derived trait representation to condition longitudinal attention across valid treatment sessions.
3. **Patient-level outcome prediction:** The attention-weighted longitudinal representation is used for binary prediction of improvement versus worsening and for complementary regression of continuous changes in depression severity.

GRU and conventional Transformer models receive the same sequences of fold-specific CNN embeddings as CoLA, enabling controlled architectural comparisons. Additional experiments use pretrained wav2vec 2.0 embeddings as an alternative speech representation.

## CoLA variants

Four conditioned session representations are evaluated:

- **Trait:** temporal session state and patient-specific trait representation;
- **Interaction:** Trait representation plus the element-wise state–trait interaction;
- **DeltaH1:** Trait representation plus change relative to the first session;
- **Full:** Trait, interaction, and baseline-relative change components.

Two conditioning sources are considered:

- **CoLA-Trait:** baseline alexithymia is directly encoded as the conditioning representation;
- **CoLA-Speech:** the conditioning representation is derived from baseline-session speech, with alexithymia used only as an auxiliary supervision target during training.

## Expected data organization

The code expects anonymized patient and session identifiers, clinical outcome variables, and paths to the speech representations.

| Field | Description |
| --- | --- |
| `Patient_ID` | Anonymized patient identifier |
| `Session` | Ordered therapy-session identifier |
| `logMel_npy` | Path to a segment-level log-Mel spectrogram |
| `CNN_SegEmb_npy` | Path to a fold-specific CNN segment embedding |
| `Embedding_npy` | Path to a segment-level wav2vec 2.0 representation |
| `Alexithymia_T0` | Baseline alexithymia score |
| `Y_Binary_Classe_Delta_Y` | Binary patient-level depression outcome |
| `Y_Standardized_T1` | Continuous patient-level outcome used for regression |

Baseline alexithymia is standardized within each training fold before being provided to CoLA-Trait.

## Recommended execution order

1. Prepare the clinical variables and longitudinal tables using `Data_Preprocessing/` and `Data_Processing/TargetVariables.ipynb`.
2. Train the fold-specific CNN encoders and extract segment embeddings using `Paper_TaCoLA/CNN_Extractor.ipynb`.
3. Run the GRU and conventional Transformer baselines using `Paper_TaCoLA/BASE_BIN.ipynb` and `Paper_TaCoLA/BASE_REG.ipynb`.
4. Run the CoLA classification and regression experiments using `Paper_TaCoLA/COLA_BIN.ipynb` and `Paper_TaCoLA/COLA_REG.ipynb`.
5. Aggregate performance metrics using `Paper_TaCoLA/Main_Metrics.ipynb`.
6. Evaluate results across the predefined random seeds and perform paired statistical analyses using `Paper_TaCoLA/Seeds_Metrics.ipynb`.

## Reproducibility and evaluation

The experimental protocol includes:

- patient-independent LOPO validation;
- fold-specific CNN training without access to the held-out patient;
- fold-wise standardization of baseline alexithymia;
- a fixed binary decision threshold of 0.5;
- random seeds 42, 123, and 2026;
- paired patient-level bootstrap confidence intervals;
- McNemar tests for paired classification decisions;
- Brier scores for probabilistic prediction quality.

The principal CoLA-Trait settings are documented in `config.yaml`. File-system paths must be adapted to the local environment.

## How to cite
[TO DE ADDED]


## Ethical and data-access notes

This repository does not contain raw clinical audio, questionnaire responses, or directly identifiable patient information. Access to non-identifiable derived features may be considered upon reasonable request and remains subject to ethical, institutional, and data-governance requirements.

## License

This software is released under the [MIT License](LICENSE).
