"""
================================================================================
ECG Signal Processing - Simple Course for Undergraduate Students
================================================================================

Created by: Torabi Signals LTD, United Kingdom
Email: torabisignals@gmail.com

Copyright © 2025 Torabi Signals LTD. All rights reserved.
This educational material is licensed and proprietary to Torabi Signals LTD.

For licensing inquiries, please contact: torabisignals@gmail.com
================================================================================

ALL-IN-ONE WORKSHOP SCRIPT FOR SYNTHETIC ECG DATA
Complete ECG Signal Processing Pipeline
Generates and processes synthetic ECG data through all 4 steps
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

def generate_synthetic_ecg(duration=60, fs=256, hr=75, noise_level=0.0):
    """Generate synthetic ECG signal with realistic P-QRS-T complexes"""
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Initialize signal
    ecg = np.zeros(n_samples)
    
    # Heart rate parameters with slight variability
    base_rr_interval = 60/hr  # seconds per beat
    
    current_pos = 0
    while current_pos < n_samples:
        # Add slight heart rate variability (±5%)
        rr_variation = np.random.uniform(0.95, 1.05)
        samples_per_beat = int(base_rr_interval * fs * rr_variation)
        beat_start = current_pos
        
        if beat_start + samples_per_beat > n_samples:
            break
            
        # P wave (atrial depolarization)
        p_peak = beat_start + int(0.08 * fs)
        if p_peak < n_samples:
            p_width = int(0.04 * fs)
            p_samples = np.arange(max(0, p_peak - p_width), min(n_samples, p_peak + p_width))
            if len(p_samples) > 0:
                ecg[p_samples] += 0.15 * np.exp(-((p_samples - p_peak)**2) / (2 * (p_width/3)**2))
        
        # Q wave (start of ventricular depolarization)
        q_peak = beat_start + int(0.14 * fs)
        if q_peak < n_samples:
            q_width = int(0.02 * fs)
            q_samples = np.arange(max(0, q_peak - q_width), min(n_samples, q_peak + q_width))
            if len(q_samples) > 0:
                ecg[q_samples] -= 0.1 * np.exp(-((q_samples - q_peak)**2) / (2 * (q_width/3)**2))
        
        # R wave (main QRS peak)
        r_peak = beat_start + int(0.18 * fs)
        if r_peak < n_samples:
            r_width = int(0.03 * fs)
            r_samples = np.arange(max(0, r_peak - r_width), min(n_samples, r_peak + r_width))
            if len(r_samples) > 0:
                ecg[r_samples] += 1.2 * np.exp(-((r_samples - r_peak)**2) / (2 * (r_width/3)**2))
        
        # S wave (end of ventricular depolarization)
        s_peak = beat_start + int(0.22 * fs)
        if s_peak < n_samples:
            s_width = int(0.02 * fs)
            s_samples = np.arange(max(0, s_peak - s_width), min(n_samples, s_peak + s_width))
            if len(s_samples) > 0:
                ecg[s_samples] -= 0.15 * np.exp(-((s_samples - s_peak)**2) / (2 * (s_width/3)**2))
        
        # T wave (ventricular repolarization)
        t_peak = beat_start + int(0.38 * fs)
        if t_peak < n_samples:
            t_width = int(0.08 * fs)
            t_samples = np.arange(max(0, t_peak - t_width), min(n_samples, t_peak + t_width))
            if len(t_samples) > 0:
                ecg[t_samples] += 0.3 * np.exp(-((t_samples - t_peak)**2) / (2 * (t_width/3)**2))
        
        current_pos += samples_per_beat
    
    # Add baseline
    ecg += 0.5
    
    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(ecg), n_samples)
        ecg += noise
    
    return ecg, t

print("="*70)
print("SYNTHETIC ECG SIGNAL PROCESSING WORKSHOP - COMPLETE PIPELINE")
print("="*70)

# ============================================
# STEP 1: GENERATE SYNTHETIC ECG DATA
# ============================================
print("\n[STEP 1] Generating synthetic ECG signal...")

# Parameters
fs = 256           # Sampling frequency (Hz)
duration = 60      # Duration (seconds)
hr = 75           # Target heart rate (BPM)

# Generate clean synthetic ECG
signal, t = generate_synthetic_ecg(duration=duration, fs=fs, hr=hr, noise_level=0.0)

print(f"✓ Successfully generated synthetic ECG")
print(f"  Duration: {duration} seconds")
print(f"  Samples: {len(signal)}")
print(f"  Sampling frequency: {fs} Hz")
print(f"  Target heart rate: {hr} BPM")

# ============================================
# STEP 2: ADD NOISE AND FILTER
# ============================================
print("\n[STEP 2] Adding noise and filtering...")

# Add noise
noise_level = 0.15
noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
noisy_signal = signal + noise

# Create bandpass filter (0.5-40 Hz for ECG)
lowcut = 0.5
highcut = 40
nyquist = fs / 2
low = lowcut / nyquist
high = highcut / nyquist

b, a = butter(4, [low, high], btype='bandpass')
filtered_signal = filtfilt(b, a, noisy_signal)

print(f"✓ Noise added and filtered ({lowcut}-{highcut} Hz)")

# ============================================
# STEP 3: DETECT R PEAKS
# ============================================
print("\n[STEP 3] Detecting R peaks...")

# Peak detection parameters
min_distance = int(0.4 * fs)
signal_75th = np.percentile(filtered_signal, 75)
signal_std = np.std(filtered_signal)
threshold = signal_75th + 0.3 * signal_std

# Detect peaks
peaks, _ = find_peaks(filtered_signal, 
                     distance=min_distance, 
                     height=threshold,
                     prominence=0.3*signal_std)

print(f"✓ Detected {len(peaks)} R peaks")

if len(peaks) > 1:
    avg_hr = len(peaks) / duration * 60
    print(f"  Average heart rate: {avg_hr:.1f} BPM")

# ============================================
# STEP 4: CALCULATE HEART RATE
# ============================================
print("\n[STEP 4] Calculating heart rate and HRV...")

if len(peaks) < 2:
    print("✗ Error: Not enough peaks for analysis")
    exit(1)

# Calculate RR intervals and heart rate
rr_intervals = np.diff(peaks) / fs
rr_intervals_ms = rr_intervals * 1000
heart_rate = 60 / rr_intervals
time_hr = peaks[1:] / fs

# HRV metrics
successive_diffs = np.diff(rr_intervals_ms)
rmssd = np.sqrt(np.mean(successive_diffs**2))
nn50 = np.sum(np.abs(successive_diffs) > 50)
pnn50 = (nn50 / len(successive_diffs)) * 100

print(f"✓ Heart rate calculated")

# ============================================
# RESULTS SUMMARY
# ============================================
print("\n" + "="*70)
print("FINAL RESULTS - SYNTHETIC ECG ANALYSIS")
print("="*70)
print(f"Recording Duration: {duration:.1f} seconds")
print(f"Number of Heartbeats: {len(peaks)}")
print(f"\nHEART RATE:")
print(f"  Average: {np.mean(heart_rate):.1f} BPM")
print(f"  Range: {np.min(heart_rate):.1f} - {np.max(heart_rate):.1f} BPM")
print(f"  Std Dev: {np.std(heart_rate):.1f} BPM")
print(f"\nHEART RATE VARIABILITY:")
print(f"  SDNN: {np.std(rr_intervals_ms):.0f} ms")
print(f"  RMSSD: {rmssd:.1f} ms")
print(f"  pNN50: {pnn50:.1f}%")
print(f"  Mean RR: {np.mean(rr_intervals_ms):.0f} ms")
print("="*70)

# ============================================
# COMPREHENSIVE VISUALIZATION
# ============================================
print("\nGenerating comprehensive visualization...")

fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

# Plot 1: Original ECG (5 seconds)
ax1 = fig.add_subplot(gs[0, 0])
plot_5s = int(5 * fs)
ax1.plot(t[:plot_5s], signal[:plot_5s], 'b-', linewidth=1.5)
ax1.set_title('Step 1: Synthetic ECG (First 5s)', fontweight='bold', fontsize=11)
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Noisy ECG (5 seconds)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t[:plot_5s], noisy_signal[:plot_5s], 'r-', linewidth=1, alpha=0.8)
ax2.set_title('Noisy ECG (First 5s)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Amplitude', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Filtered ECG (5 seconds)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t[:plot_5s], filtered_signal[:plot_5s], 'g-', linewidth=1.5)
ax3.set_title('Step 2: Filtered ECG (First 5s)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Amplitude', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison overlay (3 seconds)
ax4 = fig.add_subplot(gs[1, 1])
plot_3s = int(3 * fs)
ax4.plot(t[:plot_3s], signal[:plot_3s], 'b-', linewidth=2, label='Original', alpha=0.7)
ax4.plot(t[:plot_3s], noisy_signal[:plot_3s], 'r-', linewidth=1, label='Noisy', alpha=0.5)
ax4.plot(t[:plot_3s], filtered_signal[:plot_3s], 'g-', linewidth=2, label='Filtered', alpha=0.8)
ax4.set_title('Signal Comparison (First 3s)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Amplitude', fontsize=10)
ax4.legend(fontsize=8, loc='upper right')
ax4.grid(True, alpha=0.3)

# Plot 5: ECG with detected peaks (10 seconds)
ax5 = fig.add_subplot(gs[2, :])
plot_10s = int(10 * fs)
peaks_10s = peaks[peaks < plot_10s]
ax5.plot(t[:plot_10s], filtered_signal[:plot_10s], 'b-', linewidth=1.5, alpha=0.8)
ax5.plot(t[peaks_10s], filtered_signal[peaks_10s], 'ro', markersize=10, 
         markeredgewidth=2, label=f'R Peaks (n={len(peaks_10s)})')
ax5.axhline(y=threshold, color='g', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='Threshold')
ax5.set_title('Step 3: R Peak Detection (First 10 seconds)', fontweight='bold', fontsize=12)
ax5.set_ylabel('Amplitude', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Complete ECG with all peaks
ax6 = fig.add_subplot(gs[3, :])
ax6.plot(t, filtered_signal, 'b-', linewidth=0.8, alpha=0.7)
ax6.plot(t[peaks], filtered_signal[peaks], 'ro', markersize=4, label=f'All Peaks (n={len(peaks)})')
ax6.set_title('Complete ECG with All Detected R Peaks', fontweight='bold', fontsize=12)
ax6.set_xlabel('Time (seconds)', fontsize=10)
ax6.set_ylabel('Amplitude', fontsize=10)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Plot 7: Heart Rate Over Time
ax7 = fig.add_subplot(gs[4, 0])
ax7.plot(time_hr, heart_rate, 'r-', linewidth=2, marker='o', markersize=4)
ax7.axhline(y=np.mean(heart_rate), color='blue', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(heart_rate):.1f} BPM')
ax7.fill_between(time_hr, 
                np.mean(heart_rate) - np.std(heart_rate),
                np.mean(heart_rate) + np.std(heart_rate),
                alpha=0.2, color='blue', label='±1 SD')
ax7.set_title('Step 4: Heart Rate Over Time', fontweight='bold', fontsize=11)
ax7.set_xlabel('Time (seconds)', fontsize=10)
ax7.set_ylabel('Heart Rate (BPM)', fontsize=10)
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Heart Rate Histogram
ax8 = fig.add_subplot(gs[4, 1])
ax8.hist(heart_rate, bins=20, color='coral', edgecolor='black', alpha=0.7)
ax8.axvline(x=np.mean(heart_rate), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(heart_rate):.1f} BPM')
ax8.set_title('Heart Rate Distribution', fontweight='bold', fontsize=11)
ax8.set_xlabel('Heart Rate (BPM)', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis='y')

plt.suptitle('Complete Synthetic ECG Analysis - All Steps', fontsize=15, fontweight='bold', y=0.998)

plt.savefig('synthetic_ecg_workshop_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization complete")

# ============================================
# EXPORT RESULTS
# ============================================
print("\nExporting results...")

# Save detailed results
results = np.column_stack([
    peaks[1:],
    time_hr,
    rr_intervals_ms,
    heart_rate
])

np.savetxt('synthetic_ecg_results.csv',
          results,
          delimiter=',',
          header='Peak_Index,Time_sec,RR_Interval_ms,Heart_Rate_BPM',
          comments='',
          fmt='%.2f')

# Save summary
with open('synthetic_ecg_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("SYNTHETIC ECG ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(f"Duration: {duration:.1f} seconds\n")
    f.write(f"Heartbeats: {len(peaks)}\n")
    f.write(f"Sampling Frequency: {fs} Hz\n")
    f.write(f"Target Heart Rate: {hr} BPM\n")
    f.write(f"\nAverage HR: {np.mean(heart_rate):.1f} BPM\n")
    f.write(f"HR Range: {np.min(heart_rate):.1f} - {np.max(heart_rate):.1f} BPM\n")
    f.write(f"SDNN: {np.std(rr_intervals_ms):.0f} ms\n")
    f.write(f"RMSSD: {rmssd:.1f} ms\n")
    f.write(f"pNN50: {pnn50:.1f}%\n")
    f.write("="*70 + "\n")

print("✓ Results saved:")
print("  - synthetic_ecg_results.csv")
print("  - synthetic_ecg_summary.txt")
print("  - synthetic_ecg_workshop_complete.png")

# ============================================
# COMPLETION MESSAGE
# ============================================
print("\n" + "="*70)
print("🎉 WORKSHOP COMPLETE! 🎉")
print("="*70)
print("Successfully analyzed SYNTHETIC ECG data:")
print(f"  ✓ Generated {duration:.1f}s of synthetic ECG signal")
print(f"  ✓ Applied noise filtering (0.5-40 Hz)")
print(f"  ✓ Detected {len(peaks)} R peaks accurately")
print(f"  ✓ Calculated heart rate: {np.mean(heart_rate):.1f} ± {np.std(heart_rate):.1f} BPM")
print(f"  ✓ Analyzed HRV: SDNN = {np.std(rr_intervals_ms):.0f} ms")
print(f"  ✓ Generated comprehensive visualizations")
print(f"  ✓ Exported results for further analysis")
print("="*70)
print("\nThis synthetic ECG shows:")
if np.mean(heart_rate) < 60:
    print("  • Bradycardia (HR < 60 BPM)")
elif np.mean(heart_rate) > 100:
    print("  • Tachycardia (HR > 100 BPM)")
else:
    print("  • Normal sinus rhythm (60-100 BPM)")

if np.std(rr_intervals_ms) < 50:
    print("  • Low heart rate variability")
elif np.std(rr_intervals_ms) > 100:
    print("  • High heart rate variability")
else:
    print("  • Normal heart rate variability")
print("="*70)
