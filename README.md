# Barbell Exercise Tracking with Wearable Sensors

A machine learning pipeline that classifies barbell exercises and counts repetitions using accelerometer and gyroscope data from a MetaMotion wearable sensor.

---

## Overview

This project processes raw motion sensor data to:
1. **Classify** which barbell exercise is being performed (bench press, squat, deadlift, overhead press, barbell row, or rest)
2. **Count repetitions** for each exercise automatically

The pipeline goes from raw CSV sensor files all the way to trained classification models, using signal processing and feature engineering techniques common in sports science and health wearables.

---

## Exercises Tracked

| Exercise | Label |
|---|---|
| Bench Press | `bench` |
| Back Squat | `squat` |
| Deadlift | `dead` |
| Overhead Press | `ohp` |
| Barbell Row | `row` |
| Rest | `rest` |

---

## Project Structure

```
tracking-barbell-exercises/
├── data/
│   ├── raw/MetaMotion/        # 187 raw CSV sensor files
│   ├── interim/               # Intermediate pickle files between pipeline stages
│   └── processed/             # Final processed data
├── src/
│   ├── data/
│   │   └── make_dataset.py    # Load, merge, and resample raw sensor files
│   ├── features/
│   │   ├── build_features.py      # Full feature engineering pipeline
│   │   ├── count_repetitions.py   # Rep counting via peak detection
│   │   ├── remove_outliers.py     # Outlier detection (IQR, Chauvenet, LOF)
│   │   ├── DataTransformation.py  # Butterworth low-pass filter & PCA
│   │   ├── TemporalAbstraction.py # Rolling window statistics
│   │   └── FrequencyAbstraction.py# FFT-based frequency features
│   ├── models/
│   │   ├── train_model.py         # Model training, evaluation, comparison
│   │   ├── predict_model.py       # Run predictions on new data
│   │   └── LearningAlgorithms.py  # Wrappers for 6 ML classifiers
│   └── visualization/
│       ├── visualize.py           # Plot raw sensor data per exercise
│       └── plot_settings.py       # Shared plot styling
└── reports/
    └── figures/               # 23 sensor data visualizations
```

---

## Pipeline

### Step 1 — Data Collection & Merging (`make_dataset.py`)
- 187 CSV files (accelerometer at 12.5 Hz + gyroscope at 25 Hz)
- Metadata (participant, exercise, weight category) extracted from filenames
- Data merged and resampled to a common 200ms interval
- Output: `data/interim/01_data_processed.pkl`

### Step 2 — Outlier Removal (`remove_outliers.py`)
- Compares IQR, Chauvenet's Criterion, and Local Outlier Factor (LOF)
- **Chauvenet's Criterion** selected — handles per-exercise outlier thresholds well
- Output: `data/interim/02_outliers_removed_chauvenets.pkl`

### Step 3 — Feature Engineering (`build_features.py`)
A multi-stage feature extraction process:

| Stage | Method | Description |
|---|---|---|
| 1 | Interpolation | Fill NaN values after outlier removal |
| 2 | Butterworth Low-Pass Filter | Removes high-frequency noise (cutoff 1.2 Hz) |
| 3 | PCA | Reduces 6 raw axes to 3 principal components |
| 4 | Magnitude | `acc_r` and `gyr_r` — rotation-invariant features |
| 5 | Temporal Abstraction | Rolling mean & std (window = 5 samples) |
| 6 | Frequency Abstraction | FFT amplitudes, weighted frequency, power spectral entropy |
| 7 | K-Means Clustering | 5 clusters on acceleration data as a categorical feature |

Output: `data/interim/03_data_features.pkl` (~50+ features)

### Step 4 — Model Training (`train_model.py`)
Five feature sets (from basic to full) tested across 6 classifiers:

**Classifiers:**
- Random Forest
- Neural Network (MLP)
- K-Nearest Neighbors
- Decision Tree
- Naive Bayes
- Support Vector Machine (linear & RBF kernel)

**Evaluation:**
- 75/25 stratified train/test split
- Grid search with 5-fold cross-validation
- Participant-based split for real-world generalization testing

### Step 5 — Repetition Counting (`count_repetitions.py`)
Each exercise uses a tuned low-pass filter + peak detection algorithm:

| Exercise | Signal | Cutoff Frequency |
|---|---|---|
| Bench Press | `acc_r` | 0.4 Hz |
| Back Squat | `acc_r` | 0.35 Hz |
| Deadlift | `acc_r` | 0.4 Hz |
| Overhead Press | `acc_r` | 0.35 Hz |
| Barbell Row | `gyr_x` | 0.65 Hz |

---

## Results

- **Best Classifier**: Random Forest with full feature set (frequency + temporal + clustering)
- **Rep Counting**: Per-exercise peak detection with Mean Absolute Error (MAE) measured against ground truth (5 reps for heavy sets, 10 for medium)
- Confusion matrices generated for both random and participant-based splits

---

## Setup

### Requirements
- Python 3.8.15
- Conda environment

### Install
```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

### Run the Pipeline
```bash
# 1. Build the dataset
python src/data/make_dataset.py

# 2. Remove outliers
python src/features/remove_outliers.py

# 3. Build features
python src/features/build_features.py

# 4. Train models
python src/models/train_model.py

# 5. Count repetitions
python src/features/count_repetitions.py
```

---

## Data

Raw sensor data collected from a **MetaWear** wrist/wrist sensor worn during barbell training sessions.
- **Participants**: 5 (labeled A–E)
- **Weight categories**: Heavy, Medium
- **Total files**: 187 CSVs

---

## Tech Stack

- **Python 3.8** | Pandas | NumPy
- **Signal Processing**: SciPy (Butterworth filter, FFT)
- **Machine Learning**: scikit-learn
- **Visualization**: Matplotlib

---

## References

This project follows the structure from [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).
