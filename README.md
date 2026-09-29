# EEG Emotion Classification

Subject-independent emotion recognition from EEG. The first milestone uses the
preprocessed **DEAP** dataset to classify **low vs. high valence** from 4-second
EEG windows. Later experiments can add arousal, EEGNet, TSception, EEG
Conformer, FACED, and cross-dataset evaluation.

## First milestone

1. Inspect the DEAP files and verify their shapes.
2. Keep only the first 32 EEG channels.
3. Split participants into train/validation/test groups **before windowing**.
4. Create 4-second windows at 128 Hz.
5. Train a small 1D CNN baseline.
6. Evaluate on participants unseen during training.

The fixed development split is 24/4/4 participants. Final results should use
subject-wise cross-validation.

## Project structure

```text
eeg-emotion-classification/
├── configs/deap.yaml
├── data/raw/deap/            # put s01.dat ... s32.dat here
├── notebooks/01_deap_exploration.ipynb
├── scripts/inspect_deap.py
├── src/eeg_emotion/
│   ├── data.py
│   ├── datasets.py
│   ├── preprocessing.py
│   ├── splits.py
│   └── models/cnn.py
├── tests/
├── checkpoints/
└── results/
```

## Setup in VS Code

Open the project folder in VS Code, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate       # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Select the `.venv` Python interpreter in VS Code. The included extension file
recommends the Python and Jupyter extensions.

## Add DEAP

Download the preprocessed Python version of DEAP. Copy these files into
`data/raw/deap/`:

```text
s01.dat
s02.dat
...
s32.dat
```

Do not commit the files to GitHub. They are ignored by `.gitignore`.

Inspect the first participant:

```bash
python scripts/inspect_deap.py --data-dir data/raw/deap --subject 1
```

Expected preprocessed shapes are normally:

- data: `(40 trials, 40 channels, 8064 samples)`
- labels: `(40 trials, 4 ratings)`
- EEG input used here: first 32 channels

Then open `notebooks/01_deap_exploration.ipynb` and run the cells in order.

## Run checks

```bash
pytest
```

## Important evaluation rule

Never randomly split 4-second windows. Split by `subject_id` first; otherwise,
near-identical windows from the same person can appear in both training and
test data and produce misleading results.

