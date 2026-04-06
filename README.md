# 🎵 Music Instrument Identifier

A machine learning project that identifies musical instruments from audio samples using the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth). The project explores two approaches: a **Convolutional Neural Network (CNN)** trained on mel spectrograms, and classical **supervised learning** classifiers.

---

## 📁 Project Structure

```
music-instrument-identifier/
│
├── AnalyzeNsynthDataSet.ipynb      # Exploratory analysis of the NSynth dataset
├── analyzeNSynthWaveFile.ipynb     # Waveform-level analysis of audio samples
├── DataPreProcessing.ipynb         # Feature extraction and data preparation
├── CNN_Spectro.ipynb               # CNN model trained on mel spectrograms
├── SupervisedLearning.ipynb        # Classical ML classifiers (SVM, RF, etc.)
└── Pickles/                        # Saved models and preprocessed data
```

---

## 🧠 Approach

### 1. Data Analysis
The NSynth dataset contains annotated musical notes from a wide range of instruments. Initial notebooks explore the dataset's structure, instrument families, pitch distributions, and audio waveform characteristics.

### 2. Preprocessing
Audio samples are transformed into features suitable for machine learning:
- **Mel spectrograms** — 2D time-frequency representations fed into the CNN
- **Handcrafted features** — MFCCs, chroma, spectral features used for classical classifiers

### 3. Model Training

| Model | Input | Notebook |
|---|---|---|
| Convolutional Neural Network | Mel Spectrograms | `CNN_Spectro.ipynb` |
| Supervised Learning (SVM / RF / etc.) | Audio Features | `SupervisedLearning.ipynb` |

---

## 🗃️ Dataset

This project uses the **[NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth)** by Google Magenta — a large-scale, annotated dataset of musical notes covering 1,006 instruments across 11 instrument families.

> Download the dataset separately and update the file paths in the notebooks accordingly.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib librosa scikit-learn tensorflow keras jupyter
```

### Run the Notebooks

Clone the repository and launch Jupyter:

```bash
git clone https://github.com/ritanshudeshmukh/music-instrument-identifier.git
cd music-instrument-identifier
jupyter notebook
```

Then run the notebooks in order:

1. `AnalyzeNsynthDataSet.ipynb` — understand the data
2. `analyzeNSynthWaveFile.ipynb` — inspect audio samples
3. `DataPreProcessing.ipynb` — extract features and save preprocessed data
4. `CNN_Spectro.ipynb` — train the CNN model
5. `SupervisedLearning.ipynb` — train and evaluate classical ML models

---

## 🔧 Dependencies

- Python 3.x
- NumPy / Pandas
- Librosa (audio processing)
- Matplotlib / Seaborn (visualization)
- Scikit-learn (classical ML)
- TensorFlow / Keras (CNN)
- Jupyter Notebook

---

## 📌 Notes

- Pre-trained models and preprocessed arrays are stored as `.pkl` files in the `Pickles/` directory.
- Update dataset paths in each notebook before running.

---

## 📄 License

This project is open source. Feel free to fork, explore, and build on it.
