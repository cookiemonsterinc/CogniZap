from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import mne
import numpy as np
import joblib
from mne.time_frequency import psd_array_welch
import time
from collections import deque
import json
from datetime import datetime
import os
import sys

# Suppress warnings
mne.set_log_level("ERROR")
import warnings
warnings.filterwarnings('ignore')

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# ============================================
# CONFIGURATION
# ============================================
SERIAL_PORT = "COM2"  # Adjust as needed
SAMPLING_FREQ = 250
WINDOW_SECONDS = 2
WINDOW_SAMPLES = int(SAMPLING_FREQ * WINDOW_SECONDS)
CHANNEL_INDICES = [5, 3, 7, 8]
CH_NAMES = ['AF7', 'AF8', 'TP9', 'TP10']
CH_TYPES = ['eeg'] * len(CH_NAMES)

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# ============================================
# LOAD MODEL & SCALER
# ============================================
try:
    scaler = joblib.load(r"C:\Users\User\OneDrive\Desktop\Python\scaler_fatigue.pkl")
    model = joblib.load(r"C:\Users\User\OneDrive\Desktop\Python\svm_fatigue_model.pkl")
    print("[OK] Model & scaler loaded\n")
except Exception as e:
    print(f"[ERROR] Loading model: {e}")
    exit(1)

# ============================================
# INITIALIZE BOARD
# ============================================
try:
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    print("[OK] Board connected\n")
    time.sleep(2)
except Exception as e:
    print(f"[ERROR] Board init: {e}")
    exit(1)

# ============================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================
def get_bandpowers(raw_data, sfreq):
    rel = {}
    ab = {}
    try:
        psd_all, _ = psd_array_welch(raw_data, sfreq, fmin=1, fmax=40, n_jobs=1)
        total_pwr = np.sum(psd_all, axis=1, keepdims=True) + 1e-12
        for band_name, (fmin, fmax) in BANDS.items():
            psds, _ = psd_array_welch(raw_data, sfreq, fmin=fmin, fmax=fmax, n_jobs=1)
            rel[band_name] = (np.mean(psds / total_pwr, axis=1), np.std(psds / total_pwr, axis=1))
            ab[band_name] = (np.mean(psds, axis=1), np.std(psds, axis=1))
        return rel, ab
    except:
        return None, None

def extract_features(rel, ab):
    if rel is None or ab is None:
        return None, None

    a_m, a_s = np.mean(rel['alpha'][0]), np.mean(rel['alpha'][1])
    b_m, b_s = np.mean(rel['beta'][0]), np.mean(rel['beta'][1])
    d_m, d_s = np.mean(rel['delta'][0]), np.mean(rel['delta'][1])
    g_m, g_s = np.mean(rel['gamma'][0]), np.mean(rel['gamma'][1])
    t_m, t_s = np.mean(rel['theta'][0]), np.mean(rel['theta'][1])

    Aa_m, Aa_s = np.mean(ab['alpha'][0]), np.mean(ab['alpha'][1])
    Ab_m, Ab_s = np.mean(ab['beta'][0]), np.mean(ab['beta'][1])
    Ad_m, Ad_s = np.mean(ab['delta'][0]), np.mean(ab['delta'][1])
    Ag_m, Ag_s = np.mean(ab['gamma'][0]), np.mean(ab['gamma'][1])
    At_m, At_s = np.mean(ab['theta'][0]), np.mean(ab['theta'][1])

    theta_alpha_ratio = t_m / (a_m + 1e-12)
    alpha_beta_ratio = a_m / (b_m + 1e-12)
    theta_beta_ratio = t_m / (b_m + 1e-12)
    delta_theta_ratio = d_m / (t_m + 1e-12)
    alpha_variability = a_s / (a_m + 1e-12)
    theta_variability = t_s / (t_m + 1e-12)
    beta_variability = b_s / (b_m + 1e-12)
    delta_variability = d_s / (d_m + 1e-12)
    gamma_variability = g_s / (g_m + 1e-12)
    ab_mean = (Ab_m + Ag_m + Ad_m + Aa_m) / 4
    rel_mean = (a_m + g_m + d_m + b_m) / 4
    ab_gamma_alpha_ratio = Ag_m / (Aa_m + 1e-12)
    rel_alpha_theta_ratio = a_m / (t_m + 1e-12)

    features = np.array([
        a_m, a_s, b_m, b_s, d_m, d_s, g_m, g_s, t_m, t_s,
        Aa_m, Aa_s, Ab_m, Ab_s, Ad_m, Ad_s, Ag_m, Ag_s, At_m, At_s,
        0, 0,
        theta_alpha_ratio, alpha_beta_ratio, theta_beta_ratio, delta_theta_ratio,
        alpha_variability, theta_variability, beta_variability, delta_variability,
        gamma_variability, ab_mean, rel_mean, ab_gamma_alpha_ratio, rel_alpha_theta_ratio
    ]).reshape(1, -1)

    return features, theta_alpha_ratio

# ============================================
# CALIBRATION PHASE
# ============================================
clear_screen()
print("=" * 70)
print("🎯 FATIGUE DETECTION SYSTEM - CALIBRATION MODE")
print("=" * 70)
print()

subject_id = input("📋 Enter Subject ID (e.g., S001): ").strip()
print()

print("=" * 70)
print(f"🔍 CALIBRATING FOR SUBJECT: {subject_id}")
print("=" * 70)
print()
print(f"⏱  Phase 1: RESTING STATE (Eyes Open) - 3 minutes")
print("Please sit quietly and relax. Look at a fixed point.")
print()

input("Press ENTER when ready to start baseline calibration...")
print()

calibration_readings = []
calibration_start_time = time.time()
calibration_duration = 180  # 3 minutes

print("Calibration in progress...")
print()

while (time.time() - calibration_start_time) < calibration_duration:
    elapsed = int(time.time() - calibration_start_time)

    try:
        data = board.get_current_board_data(WINDOW_SAMPLES)
        eeg = data[CHANNEL_INDICES, :]

        if eeg.shape[1] < WINDOW_SAMPLES:
            continue

        eeg = np.where(np.abs(eeg) > 1e6, 0, eeg)

        raw = mne.io.RawArray(eeg, mne.create_info(CH_NAMES, SAMPLING_FREQ, CH_TYPES), verbose=False)
        raw.filter(1, 40, fir_design='firwin', verbose=False, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5)

        rel, ab = get_bandpowers(raw.get_data(), SAMPLING_FREQ)

        if rel is None:
            continue

        features, theta_alpha = extract_features(rel, ab)

        if features is None:
            continue

        features_scaled = scaler.transform(features)
        decision_score = model.decision_function(features_scaled)[0]
        model_confidence = min(1.0, max(0, (abs(decision_score) / 2)))
        theta_alpha_score = min(1, max(0, (theta_alpha - 0.5) / 1.5))
        fatigue_index = 0.5 * model_confidence + 0.5 * theta_alpha_score
        fatigue_pct = fatigue_index * 100
        calibration_readings.append(fatigue_pct)

        # Update every 10 seconds (to slow output)
        if elapsed % 10 == 0:
            clear_screen()
            print("=" * 70)
            print("📊 CALIBRATION IN PROGRESS")
            print("=" * 70)
            print()
            print(f"⏱  Time: {elapsed}s / {calibration_duration}s")
            print(f"📈 Current reading: {fatigue_pct:.1f}%")
            print(f"📊 Readings collected: {len(calibration_readings)}")
            print()
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[CANCELLED] Calibration stopped")
        board.stop_stream()
        board.release_session()
        exit()
    except Exception:
        continue

# Sanity check for baseline range can be added here if needed

subject_baseline = np.mean(calibration_readings)
subject_std = np.std(calibration_readings)
low_threshold = subject_baseline - 10
mod_threshold = subject_baseline + 10

clear_screen()
print("=" * 70)
print("✅ CALIBRATION COMPLETE")
print("=" * 70)
print()
print(f"Subject ID: {subject_id}")
print(f"Baseline Fatigue: {subject_baseline:.1f}%")
print(f"Standard Deviation: {subject_std:.1f}%")
print()
print("📊 PERSONALIZED THRESHOLDS FOR THIS SUBJECT:")
print(f"  🟢 LOW FATIGUE:       < {low_threshold:.0f}%")
print(f"  🟡 MODERATE FATIGUE:  {low_threshold:.0f}% - {mod_threshold:.0f}%")
print(f"  🔴 HIGH FATIGUE:      > {mod_threshold:.0f}%")
print()

# Save calibration file
calibration_data = {
    "subject_id": subject_id,
    "baseline": subject_baseline,
    "std": subject_std,
    "low_threshold": low_threshold,
    "mod_threshold": mod_threshold,
    "timestamp": datetime.now().isoformat()
}

cal_filename = f"calibration_{subject_id}.json"
with open(cal_filename, "w") as f:
    json.dump(calibration_data, f, indent=2)

print(f"💾 Saved to: {cal_filename}")
print()

input("Press ENTER to start WCST monitoring...")
print()

# ============================================
# MAIN WCST MONITORING LOOP
# ============================================
pred_buffer = deque(maxlen=10)
fatigue_buffer = deque(maxlen=60)
window_count = 0
prev_smoothed = subject_baseline
last_display_time = time.time()
DISPLAY_INTERVAL = 3  # seconds

clear_screen()
print("=" * 70)
print(f"🎯 WCST FATIGUE MONITORING - Subject {subject_id}")
print("=" * 70)
print()
print("WCST is now running...")
print("Press Ctrl+C to stop")
print()

try:
    while True:
        try:
            data = board.get_current_board_data(WINDOW_SAMPLES)
            eeg = data[CHANNEL_INDICES, :]

            if eeg.shape[1] < WINDOW_SAMPLES:
                continue

            eeg = np.where(np.abs(eeg) > 1e6, 0, eeg)

            raw = mne.io.RawArray(eeg, mne.create_info(CH_NAMES, SAMPLING_FREQ, CH_TYPES), verbose=False)
            raw.filter(1, 40, fir_design='firwin', verbose=False, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5)

            rel, ab = get_bandpowers(raw.get_data(), SAMPLING_FREQ)

            if rel is None:
                continue

            features, theta_alpha = extract_features(rel, ab)

            if features is None:
                continue

            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            decision_score = model.decision_function(features_scaled)[0]

            pred_buffer.append(pred)

            model_confidence = min(1.0, max(0, (abs(decision_score) / 2)))
            theta_alpha_score = min(1, max(0, (theta_alpha - 0.5) / 1.5))

            fatigue_index = 0.5 * model_confidence + 0.5 * theta_alpha_score
            fatigue_percentage = fatigue_index * 100
            fatigue_buffer.append(fatigue_percentage)

            alpha = 0.25
            smoothed_fatigue = alpha * fatigue_percentage + (1 - alpha) * prev_smoothed
            prev_smoothed = smoothed_fatigue

            window_count += 1

            current_time = time.time()
            if current_time - last_display_time >= DISPLAY_INTERVAL:
                last_display_time = current_time

                result = "😴 FATIGUED" if round(np.mean(list(pred_buffer))) == 1 else "⚡ ALERT"

                if smoothed_fatigue < low_threshold:
                    color = "🟢"
                    status = "LOW"
                elif smoothed_fatigue < mod_threshold:
                    color = "🟡"
                    status = "MODERATE"
                else:
                    color = "🔴"
                    status = "HIGH"

                bar_filled = int(20 * (smoothed_fatigue - low_threshold) / max(1e-6, (mod_threshold - low_threshold)))
                bar_filled = max(0, min(20, bar_filled))
                bar = "█" * bar_filled + "░" * (20 - bar_filled)

                clear_screen()
                print("=" * 70)
                print(f"🎯 WCST MONITORING - Subject {subject_id}")
                print("=" * 70)
                print()
                print(f"⏱  Window: #{window_count}")
                print()
                print(f"{color} FATIGUE STATUS: {status}")
                print()
                print(f"   Smoothed Fatigue: {smoothed_fatigue:.1f}%")
                print(f"   Raw Fatigue:      {fatigue_percentage:.1f}%")
                print()
                print(f"   Progress: [{bar}]")
                print(f"   Range: {low_threshold:.0f}% to {mod_threshold:.0f}%")
                print()
                print(f"🔮 Model Status: {result}")
                print(f"📈 Model Confidence: {model_confidence*100:.1f}%")
                print(f"🧠 Theta/Alpha Ratio: {theta_alpha:.2f}")
                print()
                print("=" * 70)
                print("Updating every 3 seconds")
                print("Press Ctrl+C to stop")
                print("=" * 70)

            time.sleep(0.5)
        except Exception:
            continue

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("[✓] MONITORING STOPPED")
    print("=" * 70 + "\n")
    board.stop_stream()
    board.release_session()
    print("[✓] Board released")
    print(f"📁 Calibration file saved: {cal_filename}\n")

