# Part of the reproduction pipeline for 'A Comparison of Random Forest, XGBoost, and SVM in Predicting ATP Tennis Match Outcomes'.
# Scripts are executed in order by run_all.py in one shared namespace (they are also notebook cells: paste them in order).

import os, sys, io, json, time, hashlib, warnings, platform, joblib
from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules: matplotlib.use("Agg")           # headless when not in a notebook
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, brier_score_loss, roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
try:
    from IPython.display import display
except Exception:
    display = print

warnings.filterwarnings("ignore")
SEED = 42

# ===== CONFIGURATION (values used for every reported result) =====
START_YEAR, END_YEAR = 2000, 2026        # years loaded (Elo is warmed up from START_YEAR)
TRAIN_START_YEAR = 2010                  # training data start here
TEST_START_DATE = "2023-01-01"           # outer test years: 2023 to 2026-05-25
OUTER_YEARS = [2023, 2024, 2025, 2026]
MIN_PRIOR_MATCHES = 10                   # cold-start rule: both players need >= 10 earlier matches
DROP_RETIRED = True                      # drop RET, W/O and DEF matches
N_TUNE = 12000                           # latest rows of each fold's training data used for tuning
N_ITER = 60                              # max hyperparameter combinations per search

# ===== DATA SOURCE: fixed snapshot =====
HF_REPO = "Aneeshers/tennis-sackmann-archive"       # mirror of github.com/JeffSackmann/tennis_atp (CC BY-NC-SA 4.0)
DATA_REVISION = os.environ.get("ATP_DATA_REVISION", "")     # Hugging Face commit SHA of the dataset; empty = floating 'main' (NOT reproducible)
DATA_DIR = os.environ.get("ATP_DATA_DIR", "")               # or a local folder holding atp_matches_YYYY.csv (e.g. the archived snapshot)
RAW_FILE = os.environ.get("ATP_RAW_FILE", "")               # or one pickled DataFrame with the exact raw records used (fallback snapshot format)
DATA_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{DATA_REVISION or 'main'}/atp"
SAVE_DIR = os.environ.get("ATP_OUT", "./outputs"); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_LANG = os.environ.get("ATP_FIG_LANG", "en")

QUICK = os.environ.get("ATP_QUICK") == "1"      # smoke test only: tiny budgets, NOT the reported configuration
if QUICK:
    N_TUNE, N_ITER = 1200, 3
    print("QUICK MODE: reduced budgets; results are not the reported ones.")
print(f"Data: {RAW_FILE or ('local folder ' + DATA_DIR if DATA_DIR else DATA_URL)} | revision: {DATA_REVISION or DATA_DIR or RAW_FILE or 'NOT PINNED'} | out: {SAVE_DIR}")
