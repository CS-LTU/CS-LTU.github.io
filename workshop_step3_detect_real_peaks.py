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

WORKSHOP STEP 3: Detect R Peaks in Real ECG
Time: 15 minutes
Goal: Find the R peaks (heartbeats) in the real ECG signal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================
# LOAD DATA FROM STEP 2
# ============================================
data = np.load('workshop_data.npz')
filtered_signal = data['filtered_signal']
fs = data['fs']
t = data['t']

print("Loaded filtered real ECG signal from Step 2")
print(f"Sampling frequency: {fs} Hz")

# ============================================
# DETECT R PEAKS IN REAL ECG
# ============================================
print("\nDetecting R peaks in real ECG...")
print("Using adaptive threshold method for robust detection")

# Parameters for real ECG peak detection
min_distance = int(0.4 * fs)  # Minimum 0.4 seconds between peaks (150 BPM max)

# Adaptive threshold based on signal statistics
# For real ECG, use more robust statistics
signal_mean = np.mean(filtered_signal)
signal_std = np.std(filtered_signal)

# Use percentile-based threshold for robustness
signal_75th = np.percentile(filtered_signal, 75)
threshold = signal_75th + 0.3 * signal_std

print(f"Signal statistics:")
print(f"  Mean: {signal_mean:.4f}")
print(f"  Std: {signal_std:.4f}")
print(f"  75th percentile: {signal_75th:.4f}")
print(f"  Detection threshold: {threshold:.4f}")

# Find peaks
peaks, properties = find_peaks(filtered_signal, 
                               distance=min_distance, 
                               height=threshold,
                               prominence=0.3*signal_std)  # Add prominence for robustness

print(f"\n✓ Found {len(peaks)} R peaks")

# Calculate actual heart rate from detected peaks
if len(peaks) > 1:
    duration = len(filtered_signal) / fs
    avg_hr = len(peaks) / duration * 60
    print(f"Average heart rate: {avg_hr:.1f} BPM")
    
    # Calculate instantaneous heart rate variability
    rr_intervals = np.diff(peaks) / fs
    if len(rr_intervals) > 0:
        print(f"RR interval range: {np.min(rr_intervals)*1000:.0f}-{np.max(rr_intervals)*1000:.0f} ms")
        print(f"Heart rate range: {60/np.max(rr_intervals):.1f}-{60/np.min(rr_intervals):.1f} BPM")

# ============================================
# QUALITY CHECK
# ============================================
print("\n" + "="*50)
print("PEAK DETECTION QUALITY CHECK")
print("="*50)

if len(peaks) > 1:
    # Check inter-beat intervals for physiological plausibility
    ibi = np.diff(peaks) / fs * 1000  # in milliseconds
    
    # Physiological range: 300ms (200 BPM) to 2000ms (30 BPM)
    valid_peaks = np.sum((ibi >= 300) & (ibi <= 2000))
    
    print(f"Total peaks detected: {len(peaks)}")
    print(f"Valid inter-beat intervals: {valid_peaks}/{len(ibi)}")
    print(f"Detection quality: {valid_peaks/len(ibi)*100:.1f}%")
    
    if valid_peaks / len(ibi) < 0.9:
        print("\n⚠ Warning: Some peaks may be incorrectly detected")
        print("Consider adjusting threshold or filtering parameters")
    else:
        print("\n✓ Good peak detection quality!")
else:
    print("⚠ Not enough peaks detected for quality analysis")

print("="*50)

# ============================================
# VISUALIZE R PEAKS
# ============================================
print("\nVisualizing detected R peaks...")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# Plot 1: First 10 seconds with peaks
plot_duration_1 = 10
plot_samples_1 = int(plot_duration_1 * fs)
peaks_plot_1 = peaks[peaks < plot_samples_1]

axes[0].plot(t[:plot_samples_1], filtered_signal[:plot_samples_1], 
             'b-', linewidth=1.5, label='ECG Signal', alpha=0.8)
axes[0].plot(t[peaks_plot_1], filtered_signal[peaks_plot_1], 
             'ro', markersize=10, markeredgewidth=2, 
             label=f'R Peaks (n={len(peaks_plot_1)})')
axes[0].axhline(y=threshold, color='g', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Threshold')
axes[0].set_title('Real ECG with Detected R Peaks (First 10 seconds)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot 2: Middle 10 seconds
plot_start_2 = int(25 * fs)
plot_end_2 = int(35 * fs)
peaks_plot_2 = peaks[(peaks >= plot_start_2) & (peaks < plot_end_2)]

axes[1].plot(t[plot_start_2:plot_end_2], filtered_signal[plot_start_2:plot_end_2], 
             'b-', linewidth=1.5, alpha=0.8)
axes[1].plot(t[peaks_plot_2], filtered_signal[peaks_plot_2], 
             'ro', markersize=10, markeredgewidth=2, 
             label=f'R Peaks (n={len(peaks_plot_2)})')
axes[1].axhline(y=threshold, color='g', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Threshold')
axes[1].set_title('Real ECG with Detected R Peaks (25-35 seconds)', 
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3)

# Plot 3: Complete signal overview with peaks
axes[2].plot(t, filtered_signal, 'b-', linewidth=0.8, alpha=0.6)
axes[2].plot(t[peaks], filtered_signal[peaks], 'ro', markersize=5, 
             label=f'All R Peaks (n={len(peaks)})')
axes[2].set_title(f'Complete Real ECG with All Detected R Peaks', 
                  fontsize=13, fontweight='bold')
axes[2].set_xlabel('Time (seconds)', fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=11)
axes[2].legend(fontsize=10, loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# DETAILED PEAK ANALYSIS
# ============================================
if len(peaks) > 1:
    print("\n" + "="*50)
    print("DETAILED R PEAK ANALYSIS")
    print("="*50)
    
    # Show first 10 peak locations
    print(f"\nFirst 10 R peaks (sample indices):")
    for i, peak in enumerate(peaks[:10]):
        time_sec = peak / fs
        print(f"  Peak {i+1}: Sample {peak}, Time = {time_sec:.3f}s, Amplitude = {filtered_signal[peak]:.4f}")
    
    # Inter-beat interval statistics
    if len(peaks) > 2:
        ibi_ms = np.diff(peaks) / fs * 1000
        print(f"\nInter-beat Intervals:")
        print(f"  Mean: {np.mean(ibi_ms):.1f} ms")
        print(f"  Std: {np.std(ibi_ms):.1f} ms")
        print(f"  Min: {np.min(ibi_ms):.1f} ms")
        print(f"  Max: {np.max(ibi_ms):.1f} ms")
    
    print("="*50)

# ============================================
# SAVE DATA FOR NEXT STEP
# ============================================
np.savez('workshop_data.npz', 
         signal=data['signal'],
         filtered_signal=filtered_signal,
         peaks=peaks,
         fs=fs, 
         t=t)

print("\n✓ Step 3 Complete!")
print("R peaks detected and saved")
print("\nNext: Run workshop_step4_calculate_hr.py")
print("\n" + "="*50)
print("TIP: If peaks are missed or extra peaks detected:")
print("  • Adjust 'threshold' calculation (line 45)")
print("  • Change 'min_distance' parameter (line 39)")
print("  • Check signal filtering in Step 2")
print("="*50)
