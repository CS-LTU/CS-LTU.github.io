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

WORKSHOP STEP 2: Add Noise and Filter Real ECG Signal
Time: 15 minutes
Goal: Add noise to real ECG and remove it with a bandpass filter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================
# LOAD DATA FROM STEP 1
# ============================================
data = np.load('workshop_data.npz')
signal = data['signal']
fs = data['fs']
t = data['t']

print("Loaded real ECG signal from Step 1")
print(f"Signal length: {len(signal)} samples")
print(f"Sampling frequency: {fs} Hz")

# ============================================
# ADD NOISE
# ============================================
print("\nAdding noise to real ECG signal...")

# Add random noise (Students can change this!)
noise_level = 0.15  # Lower noise level for real ECG to preserve features
noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
noisy_signal = signal + noise

print(f"Noise level: {noise_level} × signal std")
print(f"Signal-to-Noise Ratio: {20*np.log10(np.std(signal)/np.std(noise)):.1f} dB")

# ============================================
# REMOVE NOISE WITH FILTER
# ============================================
print("\nApplying bandpass filter to remove noise...")

# Bandpass filter for ECG (keeps frequencies between 0.5-40 Hz)
# For real ECG, we use wider bandwidth to preserve waveform features
lowcut = 0.5    # Low frequency cutoff (Hz) - removes baseline wander
highcut = 40    # High frequency cutoff (Hz) - removes high-freq noise

print(f"Filter design: {lowcut}-{highcut} Hz bandpass")

# Design Butterworth filter (4th order for better performance)
nyquist = fs / 2
low = lowcut / nyquist
high = highcut / nyquist

b, a = butter(4, [low, high], btype='bandpass')

# Apply filter (filtfilt for zero phase distortion)
filtered_signal = filtfilt(b, a, noisy_signal)

print("✓ Filter applied successfully!")

# ============================================
# SIGNAL QUALITY METRICS
# ============================================
print("\n" + "="*50)
print("SIGNAL QUALITY ANALYSIS")
print("="*50)
print(f"Original signal std: {np.std(signal):.4f}")
print(f"Noisy signal std: {np.std(noisy_signal):.4f}")
print(f"Filtered signal std: {np.std(filtered_signal):.4f}")
print(f"Noise reduction: {(1 - np.std(filtered_signal-signal)/np.std(noisy_signal-signal))*100:.1f}%")
print("="*50)

# ============================================
# VISUALIZE COMPARISON
# ============================================
# Plot first 5 seconds
plot_duration = 5
plot_samples = int(plot_duration * fs)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original real ECG signal
axes[0].plot(t[:plot_samples], signal[:plot_samples], 'b-', linewidth=1.5)
axes[0].set_title('Original Real ECG Signal', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Noisy signal
axes[1].plot(t[:plot_samples], noisy_signal[:plot_samples], 'r-', linewidth=1, alpha=0.8)
axes[1].set_title('Noisy Real ECG Signal', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].grid(True, alpha=0.3)

# Filtered signal
axes[2].plot(t[:plot_samples], filtered_signal[:plot_samples], 'g-', linewidth=1.5)
axes[2].set_title('Filtered Real ECG Signal (Noise Removed)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Time (seconds)', fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# DETAILED COMPARISON (Overlay)
# ============================================
print("\nShowing detailed comparison...")

fig, ax = plt.subplots(figsize=(14, 6))

# Show 3 seconds for detailed view
detail_duration = 3
detail_samples = int(detail_duration * fs)

ax.plot(t[:detail_samples], signal[:detail_samples], 
        'b-', linewidth=2, label='Original', alpha=0.7)
ax.plot(t[:detail_samples], noisy_signal[:detail_samples], 
        'r-', linewidth=1, label='Noisy', alpha=0.5)
ax.plot(t[:detail_samples], filtered_signal[:detail_samples], 
        'g-', linewidth=2, label='Filtered', alpha=0.8)

ax.set_title('Detailed Comparison: Real ECG Processing (First 3 seconds)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# SAVE DATA FOR NEXT STEP
# ============================================
np.savez('workshop_data.npz', 
         signal=signal, 
         noisy_signal=noisy_signal,
         filtered_signal=filtered_signal,
         fs=fs, 
         t=t)

print("\n✓ Step 2 Complete!")
print("Real ECG data saved with filtered signal")
print("\nNext: Run workshop_step3_detect_peaks.py")
print("\n" + "="*50)
print("NOTE: Real ECG shows natural variations:")
print("  • Baseline wander")
print("  • Respiratory variations")
print("  • Natural amplitude changes")
print("="*50)
