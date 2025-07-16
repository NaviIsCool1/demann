#!/usr/bin/env python3
"""
baseline_record.py

Collects a configurable number of EMG samples from the MyoWare Arduino stream,
calculates the average baseline, prints it, and saves it to 'baseline.txt'.
"""

import serial
import time
import statistics

# ==== CONFIGURATION ====
PORT = 'COM3'   # On Windows, use 'COM3', 'COM4', etc.
BAUD = 9600
SAMPLES = 100           # Number of samples to average for baseline
TIMEOUT = 0.1           # Serial read timeout in seconds
# =======================

def main():
    # 1) Open serial port and allow Arduino to reset
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    time.sleep(2)

    readings = []
    print(f"Collecting {SAMPLES} samples for baseline...")
    while len(readings) < SAMPLES:
        line = ser.readline().decode('ascii', 'ignore').strip()
        if not line:
            continue
        raw = line.split()[0]
        try:
            value = float(raw)
        except ValueError:
            continue
        readings.append(value)
        if len(readings) % 10 == 0:
            print(f"  {len(readings)} samples collected...")

    # 2) Close serial port
    ser.close()

    # 3) Compute baseline statistics
    baseline = statistics.mean(readings)
    stdev = statistics.stdev(readings) if len(readings) > 1 else 0.0

    # 4) Output results
    print(f"\nBaseline = {baseline:.3f} mV  (σ = {stdev:.3f} mV)")
    # Save to file for reuse
    with open('baseline.txt', 'w') as f:
        f.write(f"{baseline:.6f}\n")
    print("Saved baseline to 'baseline.txt'")

if __name__ == "__main__":
    main()
