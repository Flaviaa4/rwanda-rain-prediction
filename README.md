# Rwanda Next-Day Rain Prediction

Most agriculture in Rwanda is rain-fed, and climate change is making the timing of the rainy
seasons less predictable. This project asks whether today's weather conditions at a location
can predict whether it will rain there tomorrow, to support farmers' planting, spraying and
harvesting decisions. The task is framed as binary classification and solved with both
traditional machine learning (scikit-learn) and deep learning (TensorFlow/Keras with the
`tf.data` API), compared under identical data, features, splits and metrics.

## Dataset

* **Source:** [NASA POWER](https://power.larc.nasa.gov/) daily point API
  ([documentation](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)),
  agroclimatology community.
* **Period:** 1 January 1981 to 31 December 2025, daily. The analysis starts in 1984 because
  solar radiation and cloud cover are unavailable before then.
* **Locations:** 12 district centres were requested. Muhanga falls in the same NASA grid cell as
  Kigali and Rubavu in the same cell as Musanze, so their identical series are removed, leaving
  **10 locations**: Kigali, Musanze, Huye, Rusizi, Nyagatare, Kirehe, Nyamagabe, Rwamagana,
  Karongi and Gicumbi.
* **Parameters (14):** `PRECTOTCORR` (corrected precipitation), `T2M`, `T2M_MAX`, `T2M_MIN`,
  `T2MDEW`, `RH2M`, `QV2M`, `PS`, `WS2M`, `WS2M_MAX`, `WD2M`, `ALLSKY_SFC_SW_DWN`, `CLOUD_AMT`,
  `GWETTOP`.
* **Target:** `RainTomorrow = 1` when the next day's `PRECTOTCORR` at the same location is at
  least **1.0 mm** (a rain day), otherwise 0. About 59% of days are followed by rain.

Raw data is **not** stored in this repository. Either run `notebooks/01_download_data.ipynb`
(a few minutes, resumes after a disconnect) or download `rwanda_weather_raw.zip` from the
Google Drive folder below and extract it into the project folder.

## Repository structure

```
rwanda-rain-prediction/
├── README.md
├── requirements.txt
├── configs/
│   ├── config.yaml              
│   └── experiments.yaml        
├── src/
│   ├── config.py                
│   ├── data_download.py         
│   ├── data_prep.py             
│   ├── features.py              
│   ├── split.py                 
│   ├── baselines.py             
│   ├── metrics.py               
│   ├── ml.py                     
│   ├── dl.py                    
│   ├── runner.py                
│   ├── evaluation.py             
│   └── plots.py                 
└── notebooks/
    ├── 01_download_data.ipynb   
    └── 02_rain_prediction.ipynb 
```

Each stage passes one well-defined object to the next:

```
build_raw_dataset -> load_raw -> clean -> add_target -> build_features -> split_by_year
    -> run_experiments (baselines, ml, dl) -> select_best -> final_test_evaluation
```

Data, trained models and results are written to a project folder outside Git
(`/content/drive/MyDrive/rwanda-rain-project` in Colab, `./project_data` locally):

```
rwanda-rain-project/
├── raw/<Location>.csv           
├── rwanda_weather_raw.zip       
├── rwanda_weather_daily.csv     
├── models/                     
└── results/                     
```

## How to run

**Google Colab**

1. Open `notebooks/01_download_data.ipynb` from GitHub in Colab
   (*File > Open notebook > GitHub*) and run all cells once to build the dataset on Drive.
2. Open `notebooks/02_rain_prediction.ipynb` and choose *Runtime > Restart and run all*. The
   first cell mounts Drive, clones this repository and installs the requirements.

With `run.mode: auto` in `configs/config.yaml`, experiments already saved on Drive are reloaded
instead of retrained, so a disconnect never costs the full training time (about 40 minutes).
Set `run.mode: train` to retrain everything.

**Locally.**

```bash
git clone https://github.com/Flaviaa4/rwanda-rain-prediction.git
cd rwanda-rain-prediction
pip install -r requirements.txt
export PROJECT_DIR=/path/to/rwanda-rain-project   # optional, defaults to ./project_data
jupyter notebook notebooks/
```

Every setting (dates, thresholds, feature windows, split years, hyperparameters) lives in
`configs/`, so experiments can be changed without editing code.

## Methodology

* **Cleaning:** NASA's missing-value code is replaced, duplicate grid cells are removed, the
  analysis starts in 1984 (no imputation needed), and each location's daily sequence is
  verified to have no gaps.
* **Chronological split:** training 1984 to 2015 (116,850 rows), validation 2016 to 2020
  (18,270), test 2021 to 2025 (18,250). A random split would leak near-identical days between
  sets, because consecutive days and same-day neighbouring sites are strongly correlated.
* **Baselines:** always rain, persistence (tomorrow equals today) and monthly climatology.
* **Features:** 37 base features (raw parameters, rain lags and rolling sums, days since rain,
  day-to-day changes, temperature range, dewpoint depression, circular day of year, rainy
  season flag, circular wind direction) plus 15 regional features (today's rain at every site,
  regional humidity, cloud and pressure change, share of sites with rain) for 52 in total.
  Every feature uses only information available at the end of the current day.
* **Traditional ML:** logistic regression, linear SVM, random forest and histogram gradient
  boosting, tuned with `RandomizedSearchCV` on a time-aware `PredefinedSplit` (train on the
  training years, score on the validation years).
* **Deep learning:** a Sequential MLP improved step by step (batch normalisation, dropout, L2,
  learning rate, batch size, SGD with momentum, depth) and a Functional API model with a
  learned location embedding, all fed through `tf.data` with early stopping.
* **Evaluation:** macro F1 (primary), accuracy, precision, recall, ROC-AUC, learning curves,
  confusion matrices, ROC curves, error analysis and feature importance. The best model of each
  paradigm is chosen on validation macro F1, and the test split is used once.

## Results

Validation results (macro F1, ROC-AUC and the gap between training and validation accuracy):

| Run | Macro F1 | ROC-AUC | Gap | Run | Macro F1 | ROC-AUC | Gap |
|---|---|---|---|---|---|---|---|
| Persistence baseline | 0.787 |  |  | Climatology baseline | 0.702 |  |  |
| ml_01 logistic regression | 0.802 | 0.878 | 0.011 | dl_01 baseline | 0.797 | 0.878 | 0.028 |
| ml_02 linear SVM | 0.795 | 0.877 | 0.018 | dl_02 batch norm | 0.798 | 0.877 | 0.022 |
| ml_03 random forest | 0.800 | 0.876 | 0.124 | dl_03 dropout | 0.801 | 0.881 | 0.027 |
| ml_04 gradient boosting | 0.802 | 0.879 | 0.032 | dl_04 L2 | 0.802 | 0.879 | 0.023 |
| ml_05 logreg tuned | 0.803 | 0.878 | 0.011 | dl_05 lr 3e-4 | 0.798 | 0.880 | 0.025 |
| ml_06 random forest tuned | 0.801 | 0.876 | 0.123 | dl_06 batch 1024 | 0.800 | 0.880 | 0.030 |
| ml_07 boosting tuned | 0.802 | 0.880 | 0.029 | dl_07 SGD | 0.800 | 0.880 | 0.025 |
| **ml_08 logreg regional** | **0.804** | 0.880 | 0.014 | dl_08 deeper | 0.798 | 0.878 | 0.027 |
| ml_09 boosting regional | 0.801 | 0.881 | 0.042 | dl_09 functional | 0.800 | 0.879 | 0.027 |

Selected models: `ml_08_logreg_regional` (traditional ML) and `dl_04_l2_1e-4` (deep learning).

**Test results (2021 to 2025):** On the test years 2021 to 2025, logistic regression reached a macro F1 of 0.814 and the neural network 0.817, both above the persistence baseline (0.802).

## Links

* **Google Drive folder**
* **Video presentation:**

## Limitations

* NASA POWER values are satellite and reanalysis estimates for grid cells, not rain-gauge
  measurements at the district centres.
* Daily resolution hides when rain falls within the day.
* The target is derived from a 1 mm threshold, so days near the threshold are almost
  arbitrarily labelled.
* The test period (2021 to 2025) is slightly drier than the training period.
* Ten locations give limited spatial coverage of Rwanda's varied terrain.
