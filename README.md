# 🧠 Axon A Voice Alzheimer - Early Detection of Alzheimer's Disease Based on Voice Analysis  

![GitHub repo size](https://img.shields.io/github/repo-size/romanturowskidev/axon_a_voice_alzheimer)  
![GitHub contributors](https://img.shields.io/github/contributors/romanturowskidev/axon_a_voice_alzheimer)  
![GitHub stars](https://img.shields.io/github/stars/romanturowskidev/axon_a_voice_alzheimer?style=social)  

---

## 📌 Project Description  

**Axon A Voice Alzheimer** is a project that leverages **artificial intelligence (AI) and audio processing** to aid in the **early detection of Alzheimer's disease** based on voice analysis. The model analyzes speech samples and classifies them as healthy or potentially indicative of the disease.

🔹 **Technologies used**:  
- **Python**, **TensorFlow**, **Keras** – AI model training  
- **Librosa**, **Scikit-learn** – Audio feature extraction  
- **Flask**, **Streamlit** – API and user interface  
- **Docker** – Model deployment  

🎯 **Project Goal**: Build a high-precision AI model to assist doctors and researchers in diagnosing Alzheimer's disease through voice analysis.

---

## 📁 Project Structure
  
alzheimer-prediction/               # Root directory of the project
│
├── data/                           # Data storage
│   ├── processed/                  # Preprocessed audio features
│   ├── raw/                        # Raw audio files before processing
│   ├── external/                   # External datasets (if any)
│   └── augmented/                  # Data augmentation files (if used)
│
├── myenv_axon_a_voice_alzheimer/   # Virtual environment (ignored in .gitignore)
│
├── notebooks/                      # Jupyter for data exploration & experiments
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── feature_extraction.ipynb    # Audio feature extraction analysis
│   └── model_training.ipynb        # Model training and evaluation
│
├── scripts/                        # Automation scripts
│   ├── data_processing.py          # Script for data preprocessing
│   ├── train.py                    # Script for training the model
│   ├── evaluate.py                 # Model evaluation script
│   └── deploy.py                   # Deployment automation script
│
├── src/                            # Main source code directory
│   ├── app/                        # Web API & User Interface
│   │   ├── api.py                  # Flask API for model inference
│   │   └── streamlit_app.py        # Streamlit UI for interactive usage
│   │
│   ├── config/                     # Configuration files
│   │   ├── config.yaml             # General configuration file
│   │   └── logging.yaml            # Logging settings
│   │
│   ├── evaluation/                 # Model evaluation & explainability
│   │   ├── explainability.py       # Model interpretability (SHAP, etc.)
│   │   ├── metrics.py              # Performance metrics calculations
│   │   └── validate_model.py       # Model validation before deployment
│   │
│   ├── models/                     # Machine learning models
│   │   ├── cnn_model.py            # Convolutional Neural Network (CNN)
│   │   ├── rnn_model.py            # Recurrent Neural Network (RNN)
│   │   └── train_model.py          # Model training implementation
│   │
│   ├── utils/                      # Utility functions for data processing
│   │   ├── audio_preprocessing.py  # Audio cleaning & noise removal
│   │   ├── feature_extraction.py   # Extracting MFCC, chroma, etc.
│   │   └── model_utils.py          # Helper functions for ML models
│   │
│   ├── __init__.py                 # Marks `src/` as a Python package
│   └── main.py                     # Main entry point for the project
│
├── tests/                          # Unit and integration tests
│   ├── test_audio_preprocessing.py # Tests for audio preprocessing functions
│   ├── test_end_to_end.py          # End-to-end tests for the pipeline
│   ├── test_feature_extraction.py  # Tests for feature extraction functions
│   └── test_model_training.py      # Tests for model training scripts
│
├── .gitignore                      # Specifies files to ignore in Git control
├── docker-compose.yml              # Docker Compose configuration
├── Dockerfile                      # Instructions for building the Doc container
├── LICENSE                         # License information for the project
├── poetry.lock                     # Poetry dependency lock file
├── pyproject.toml                  # Poetry configuration file for dependencies
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies for pip
