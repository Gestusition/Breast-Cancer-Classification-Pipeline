"""
Proje Yapılandırması ve Ortak Ayarlar
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import set_config

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)
sns.set_theme(style="whitegrid")
set_config(transform_output="pandas")

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
