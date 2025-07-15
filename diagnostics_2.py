#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR   = os.getcwd()
RAW_CSV    = os.path.join(BASE_DIR, 'onset_test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'raw_mlp.h5')
WINDOW     = 10
CENTER     = WINDOW // 2
THRESHOLD  = 0.5

try:
    raw_vals = np.loadtxt(RAW_CSV, delimiter=',', skiprows=1)
except:
    raw_vals = np.loadtxt(RAW_CSV)

model = load_model(MODEL_PATH)

N = len(raw_vals)
preds = np.zeros(N, dtype=int)
for i in range(0, N - WINDOW + 1):
    window_block = raw_vals[i:i+WINDOW].reshape(1, -1)
    prob = model.predict(window_block, verbose=0)[0,0]
    preds[i + CENTER] = int(prob >= THRESHOLD)

df = pd.DataFrame({
    'mV': raw_vals,
    'pred_label': preds
})
out_csv = os.path.join(BASE_DIR, 'onset_test_predictions.csv')
df.to_csv(out_csv, index=False)
print(f"Saved predictions to {out_csv}")
