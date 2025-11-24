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

WORKSHOP STEP 4: Calculate Heart Rate from ECG
Time: 15 minutes
Goal: Calculate heart rate and analyze heart rate variability from ECG data
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# LOAD DATA FROM STEP 3
# ============================================
print("="*60)
print("STEP 4: HEART RATE CALCULATION")
print("="*60)

# Check if workshop data exists
if not os.path.exists('workshop_data.npz'):
    print("\n✗ Error: 'workshop_data.npz' not found!")
    print("\nPlease run the previous steps first:")
    print("  1. workshop_step1_load_real_ecg.py")
    print("  2. workshop_step2_filter_real_ecg.py")
    print("  3. workshop_step3_detect_real_peaks.py")
    print("\nOr run the all-in-one script:")
    print("  python workshop_synthetic_ecg_all_in_one.py")
    exit(1)

try:
    data = np.load('workshop_data.npz')
    
    # Check what keys are available
    available_keys = list(data.keys())
    print(f"\nLoading data from Step 3...")
    print(f"Available data: {', '.join(available_keys)}")
    
    # Try to load required data with fallbacks
    if 'peaks' not in available_keys:
        print("\n✗ Error: R peaks not found in workshop_data.npz")
        print("Please run Step 3 (workshop_step3_detect_real_peaks.py) first")
        exit(1)
    
    peaks = data['peaks']
    fs = int(data['fs'])
    
    # Load filtered signal (required for visualization)
    if 'filtered_signal' in available_keys:
        filtered_signal = data['filtered_signal']
    else:
        print("⚠ Warning: filtered_signal not found, using signal instead")
        filtered_signal = data['signal']
    
    # Load time array
    if 't' in available_keys:
        t = data['t']
    else:
        t = np.arange(len(filtered_signal)) / fs
    
    print(f"✓ Data loaded successfully")
    print(f"  Number of R peaks: {len(peaks)}")
    print(f"  Recording duration: {len(filtered_signal)/fs:.1f} seconds")
    print(f"  Sampling frequency: {fs} Hz")

except Exception as e:
    print(f"\n✗ Error loading data: {e}")
    print("\nPlease ensure you've run the previous workshop steps.")
    exit(1)

# ============================================
# CALCULATE HEART RATE
# ============================================
print("\nCalculating heart rate from ECG data...")

if len(peaks) < 2:
    print("✗ Error: Not enough peaks detected for heart rate calculation")
    print(f"   Only {len(peaks)} peak(s) found (need at least 2)")
    print("\nTroubleshooting:")
    print("  • Return to Step 3 and adjust detection parameters")
    print("  • Check if the ECG signal quality is good")
    print("  • Try lowering the detection threshold")
    exit(1)

# Step 1: Calculate RR intervals (time between consecutive beats)
rr_intervals = np.diff(peaks) / fs  # Convert to seconds
rr_intervals_ms = rr_intervals * 1000  # Also in milliseconds

print(f"✓ Calculated {len(rr_intervals)} RR intervals")

# Step 2: Convert RR intervals to instantaneous heart rate (BPM)
heart_rate = 60 / rr_intervals  # beats per minute

# Step 3: Calculate time points for each heart rate value
time_hr = peaks[1:] / fs  # Time of each heartbeat (excluding first)

# ============================================
# HEART RATE STATISTICS
# ============================================
print("\n" + "="*60)
print("HEART RATE ANALYSIS")
print("="*60)
print(f"Recording duration: {len(filtered_signal)/fs:.1f} seconds")
print(f"Number of heartbeats: {len(peaks)}")
print(f"\nHEART RATE STATISTICS:")
print(f"  Average Heart Rate: {np.mean(heart_rate):.1f} BPM")
print(f"  Minimum Heart Rate: {np.min(heart_rate):.1f} BPM")
print(f"  Maximum Heart Rate: {np.max(heart_rate):.1f} BPM")
print(f"  Std Deviation: {np.std(heart_rate):.1f} BPM")
print(f"  Range: {np.max(heart_rate) - np.min(heart_rate):.1f} BPM")

# ============================================
# HEART RATE VARIABILITY (HRV) ANALYSIS
# ============================================
print(f"\nRR INTERVAL STATISTICS:")
print(f"  Mean RR Interval: {np.mean(rr_intervals_ms):.0f} ms")
print(f"  Std Deviation (SDNN): {np.std(rr_intervals_ms):.0f} ms")
print(f"  Min RR Interval: {np.min(rr_intervals_ms):.0f} ms")
print(f"  Max RR Interval: {np.max(rr_intervals_ms):.0f} ms")

# Additional HRV metrics
rmssd = None
pnn50 = None

if len(rr_intervals) > 2:
    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    
    # pNN50: Percentage of successive RR intervals that differ by more than 50ms
    nn50 = np.sum(np.abs(successive_diffs) > 50)
    pnn50 = (nn50 / len(successive_diffs)) * 100
    
    print(f"\nADVANCED HRV METRICS:")
    print(f"  RMSSD: {rmssd:.1f} ms (parasympathetic activity)")
    print(f"  pNN50: {pnn50:.1f}% (high-frequency variability)")
    
    # Clinical interpretation
    print(f"\nCLINICAL CONTEXT:")
    if np.mean(heart_rate) < 60:
        print(f"  • Bradycardia (HR < 60 BPM) - may be normal for athletes")
    elif np.mean(heart_rate) > 100:
        print(f"  • Tachycardia (HR > 100 BPM)")
    else:
        print(f"  • Normal sinus rhythm (60-100 BPM)")
    
    if np.std(rr_intervals_ms) < 50:
        print(f"  • Low HRV (SDNN < 50ms) - may indicate stress")
    elif np.std(rr_intervals_ms) > 100:
        print(f"  • High HRV (SDNN > 100ms) - good cardiac health")
    else:
        print(f"  • Normal HRV (SDNN 50-100ms)")

print("="*60)

# ============================================
# COMPREHENSIVE VISUALIZATION
# ============================================
print("\nCreating comprehensive visualization...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.25)

# Plot 1: Complete ECG with R peaks (top, spanning both columns)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, filtered_signal, 'b-', linewidth=0.8, alpha=0.7, label='ECG Signal')
ax1.plot(t[peaks], filtered_signal[peaks], 'ro', markersize=4, 
         markeredgewidth=1, label=f'R Peaks (n={len(peaks)})')
ax1.set_title('Complete ECG Signal with Detected R Peaks', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Amplitude', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Heart Rate Over Time
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time_hr, heart_rate, 'r-', linewidth=2, marker='o', markersize=4, 
         markeredgewidth=0.5, markeredgecolor='darkred')
ax2.axhline(y=np.mean(heart_rate), color='blue', linestyle='--', linewidth=2,
            label=f'Average: {np.mean(heart_rate):.1f} BPM')
ax2.fill_between(time_hr, 
                 np.mean(heart_rate) - np.std(heart_rate),
                 np.mean(heart_rate) + np.std(heart_rate),
                 alpha=0.2, color='blue', label=f'±1 SD')
ax2.set_title('Instantaneous Heart Rate Over Time', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('Heart Rate (BPM)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([max(40, np.mean(heart_rate)-30), min(120, np.mean(heart_rate)+30)])

# Plot 3: RR Interval Tachogram
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(time_hr, rr_intervals_ms, 'g-', linewidth=2, marker='s', markersize=4)
ax3.axhline(y=np.mean(rr_intervals_ms), color='darkgreen', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(rr_intervals_ms):.0f} ms')
ax3.set_title('RR Interval Tachogram', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time (seconds)', fontsize=10)
ax3.set_ylabel('RR Interval (ms)', fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Heart Rate Histogram
ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(heart_rate, bins=20, color='coral', edgecolor='black', alpha=0.7)
ax4.axvline(x=np.mean(heart_rate), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(heart_rate):.1f} BPM')
ax4.set_title('Heart Rate Distribution', fontsize=12, fontweight='bold')
ax4.set_xlabel('Heart Rate (BPM)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: RR Interval Histogram
ax5 = fig.add_subplot(gs[3, 0])
ax5.hist(rr_intervals_ms, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
ax5.axvline(x=np.mean(rr_intervals_ms), color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(rr_intervals_ms):.0f} ms')
ax5.set_title('RR Interval Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('RR Interval (ms)', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Poincaré Plot (HRV visualization)
ax6 = fig.add_subplot(gs[3, 1])
if len(rr_intervals_ms) > 1:
    ax6.scatter(rr_intervals_ms[:-1], rr_intervals_ms[1:], 
               c='purple', alpha=0.6, edgecolors='black', linewidth=0.5)
    # Add identity line
    min_rr = np.min(rr_intervals_ms)
    max_rr = np.max(rr_intervals_ms)
    ax6.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', linewidth=1, alpha=0.5)
    ax6.set_title('Poincaré Plot (HRV Pattern)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('RR(n) [ms]', fontsize=10)
    ax6.set_ylabel('RR(n+1) [ms]', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')

plt.suptitle('Complete Heart Rate Analysis', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('ecg_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# EXPORT RESULTS TO CSV
# ============================================
print("\nExporting results to CSV...")

# Create comprehensive results table
results = np.column_stack([
    peaks[1:],              # Peak index
    time_hr,                # Time (seconds)
    rr_intervals_ms,        # RR interval (ms)
    heart_rate              # Heart rate (BPM)
])

# Save to CSV
np.savetxt('ecg_heart_rate_results.csv',
           results,
           delimiter=',',
           header='Peak_Index,Time_sec,RR_Interval_ms,Heart_Rate_BPM',
           comments='',
           fmt='%.2f')

# Also save summary statistics
with open('ecg_summary_statistics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("HEART RATE ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"Recording Duration: {len(filtered_signal)/fs:.1f} seconds\n")
    f.write(f"Number of Heartbeats: {len(peaks)}\n")
    f.write(f"Sampling Frequency: {fs} Hz\n")
    f.write("\nHEART RATE STATISTICS:\n")
    f.write(f"  Mean: {np.mean(heart_rate):.1f} BPM\n")
    f.write(f"  Min: {np.min(heart_rate):.1f} BPM\n")
    f.write(f"  Max: {np.max(heart_rate):.1f} BPM\n")
    f.write(f"  Std: {np.std(heart_rate):.1f} BPM\n")
    f.write("\nRR INTERVAL STATISTICS:\n")
    f.write(f"  Mean: {np.mean(rr_intervals_ms):.0f} ms\n")
    f.write(f"  SDNN: {np.std(rr_intervals_ms):.0f} ms\n")
    if rmssd is not None and pnn50 is not None:
        f.write(f"  RMSSD: {rmssd:.1f} ms\n")
        f.write(f"  pNN50: {pnn50:.1f}%\n")
    f.write("="*60 + "\n")

print("✓ Results saved to 'ecg_heart_rate_results.csv'")
print("✓ Summary saved to 'ecg_summary_statistics.txt'")
print("✓ Plot saved to 'ecg_complete_analysis.png'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("WORKSHOP COMPLETE! 🎉")
print("="*60)
print("You have successfully analyzed ECG data and:")
print("  ✓ Loaded ECG signal")
print("  ✓ Applied noise filtering")
print("  ✓ Detected R peaks accurately")
print("  ✓ Calculated heart rate and HRV metrics")
print("  ✓ Generated comprehensive visualizations")
print("  ✓ Exported results for further analysis")
print("\nKey Findings from Your ECG:")
print(f"  • Average Heart Rate: {np.mean(heart_rate):.1f} BPM")
print(f"  • Heart Rate Variability: {np.std(rr_intervals_ms):.0f} ms (SDNN)")
print(f"  • Total Heartbeats Analyzed: {len(peaks)}")
print("="*60)
print("\nNext Steps:")
print("  • Analyze the CSV file in Excel or Python")
print("  • Compare with clinical standards")
print("  • Try analyzing different ECG recordings")
print("  • Explore advanced HRV analysis techniques")
print("="*60)
