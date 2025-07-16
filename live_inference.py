#!/usr/bin/env python3
import serial
import time
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ==== CONFIG =====
PORT       = 'COM3'            # adjust to your Arduino’s COM port
BAUD       = 9600
WINDOW     = 10
CENTER     = WINDOW // 2
THRESHOLD  = 0.5
MODEL_H5   = 'raw_mlp.h5'      # your saved Keras model file
# ==================

# 1) Open serial port
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # give Arduino time to reset

# 2) Load your Keras model
model = load_model(MODEL_H5)

# 3) Prepare sliding-window buffer
buffer = deque([0.0] * WINDOW, maxlen=WINDOW)

print(f"▶ Streaming EMG on {PORT}@{BAUD}, window={WINDOW}. Ctrl-C to stop.", flush=True)

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        # strip off any unit suffix (e.g. "4565.0 mV" -> "4565.0")
        raw = line.split()[0]
        try:
            mv = float(raw)
        except ValueError:
            continue

        # slide window
        buffer.append(mv)
        window_block = np.array(buffer, dtype=np.float32).reshape(1, -1)

        # perform inference
        prob = model.predict(window_block, verbose=0)[0, 0]
        label = int(prob >= THRESHOLD)

        ts = time.time()
        print(f"{ts:.3f}\tEMG={mv:7.1f} mV\tP(onset)={prob:.3f}\tLabel={label}", flush=True)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    ser.close()
