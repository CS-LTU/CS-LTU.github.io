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

WORKSHOP STEP 1: Load Real ECG Signal
Time: 15 minutes
Goal: Load and visualize a real ECG signal from .mat file
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ============================================
# PARAMETERS
# ============================================
fs = 256           # Sampling frequency (Hz) - Real ECG data
duration = 60      # Duration to extract (seconds)

# ============================================
# LOAD REAL ECG DATA
# ============================================
print("Loading real ECG signal from .mat file...")

try:
    # Load the .mat file
    data = loadmat('ecg.mat')
    
    # Auto-detect the ECG variable name (excluding metadata keys)
    ecg_key = None
    for key in data.keys():
        if not key.startswith('__'):  # Skip metadata
            ecg_key = key
            break
    
    if ecg_key is None:
        print("Error: No data variables found in ecg.mat")
        exit(1)
    
    print(f"✓ Found variable: '{ecg_key}'")
    
    # Load and flatten the ECG data
    ecg_full = data[ecg_key].flatten()  # Flatten to 1D array
    
    print(f"✓ Successfully loaded ECG data")
    print(f"  Variable name: {ecg_key}")
    print(f"  Total samples: {len(ecg_full)}")
    print(f"  Total duration: {len(ecg_full)/fs:.1f} seconds")
    print(f"  Sampling frequency: {fs} Hz")
    
except FileNotFoundError:
    print("Error: 'ecg.mat' file not found!")
    print("Please ensure the file is in the same directory as this script.")
    exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    print("\nRun 'python check_mat_file.py' to diagnose the issue")
    exit(1)

# ============================================
# EXTRACT 60-SECOND SEGMENT
# ============================================
print(f"\nExtracting {duration}-second segment for workshop...")

# Calculate number of samples needed
samples_needed = duration * fs

# Extract first 60 seconds (or maximum available)
if len(ecg_full) >= samples_needed:
    signal = ecg_full[:samples_needed]
    print(f"✓ Extracted {duration} seconds ({len(signal)} samples)")
else:
    signal = ecg_full
    actual_duration = len(signal) / fs
    print(f"⚠ Only {actual_duration:.1f} seconds available, using all data")
    duration = actual_duration

# Create time array
t = np.arange(len(signal)) / fs

# ============================================
# SIGNAL STATISTICS
# ============================================
print("\n" + "="*50)
print("ECG SIGNAL STATISTICS")
print("="*50)
print(f"Signal length: {len(signal)} samples")
print(f"Duration: {duration:.1f} seconds")
print(f"Sampling frequency: {fs} Hz")
print(f"Mean amplitude: {np.mean(signal):.4f}")
print(f"Std deviation: {np.std(signal):.4f}")
print(f"Min value: {np.min(signal):.4f}")
print(f"Max value: {np.max(signal):.4f}")
print("="*50)

# ============================================
# VISUALIZE REAL ECG
# ============================================
print("\nVisualizing ECG signal...")

# Create figure with multiple views
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: First 5 seconds
plot_duration_1 = 5
plot_samples_1 = int(plot_duration_1 * fs)
axes[0].plot(t[:plot_samples_1], signal[:plot_samples_1], 'b-', linewidth=1.5)
axes[0].set_title('Real ECG Signal (First 5 seconds)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: 10-15 seconds
plot_start_2 = int(10 * fs)
plot_end_2 = int(15 * fs)
axes[1].plot(t[plot_start_2:plot_end_2], signal[plot_start_2:plot_end_2], 'g-', linewidth=1.5)
axes[1].set_title('Real ECG Signal (10-15 seconds)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Plot 3: Complete 60 seconds overview
axes[2].plot(t, signal, 'r-', linewidth=0.8, alpha=0.7)
axes[2].set_title(f'Complete Real ECG Signal ({duration:.0f} seconds)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time (seconds)', fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# ESTIMATE HEART RATE (Rough estimate)
# ============================================
print("\nEstimating approximate heart rate...")

# Simple peak detection for rough estimate
from scipy.signal import find_peaks

# Use a very basic threshold
threshold = np.mean(signal) + 0.5 * np.std(signal)
peaks_rough, _ = find_peaks(signal, distance=int(0.4*fs), height=threshold)

if len(peaks_rough) > 1:
    estimated_hr = len(peaks_rough) / duration * 60
    print(f"Rough estimated heart rate: {estimated_hr:.1f} BPM")
    print(f"(Based on {len(peaks_rough)} detected peaks)")
else:
    print("Could not estimate heart rate - peaks will be detected in Step 3")

# ============================================
# SAVE DATA FOR NEXT STEP
# ============================================
np.savez('workshop_data.npz', signal=signal, fs=fs, t=t)

print("\n✓ Step 1 Complete!")
print("Real ECG data loaded and saved as 'workshop_data.npz'")
print("\nNext: Run workshop_step2_add_noise.py")
print("\n" + "="*50)
print("NOTE: This is REAL ECG data, so you'll see:")
print("  • Natural heart rate variability")
print("  • Realistic waveform morphology")
print("  • Actual P-QRS-T complexes")
print("="*50)
