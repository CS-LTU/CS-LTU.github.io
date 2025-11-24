# 📊 ECG Signal Processing Workshop - Student Guide

**Created by: Torabi Signals LTD, United Kingdom**  
**Email:** torabisignals@gmail.com

---

## 🎓 Welcome to the ECG Signal Processing Workshop!

This comprehensive workshop will teach you how to process real ECG (electrocardiogram) signals step by step. You will learn to:

- ✅ Load and visualise real ECG data
- ✅ Filter noise from ECG signals
- ✅ Detect heartbeats (R peaks)
- ✅ Calculate heart rate and heart rate variability (HRV)
- ✅ Generate professional analysis reports

**Duration:** Approximately 60-90 minutes  
**Level:** Undergraduate students (beginner-friendly)  
**Prerequisites:** Basic Python knowledge

---

## 🔧 Step 1: Check Your Python Environment

### 1.1 Check Python Version

Before starting, make sure you have **Python 3.8 or higher** installed.

**Open your terminal/command prompt and type:**

```bash
python --version
```

or

```bash
python3 --version
```

✅ **You should see:** `Python 3.8.x` or higher  
❌ **If you see Python 2.x or get an error:** Please install Python from [python.org](https://www.python.org/downloads/)

---

## 📦 Step 2: Install Required Libraries

### 2.1 Required Libraries

This workshop uses the following Python libraries:

- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `scipy` - Scientific computing (signal processing)

### 2.2 Installation Commands

**Install all libraries at once:**

```bash
pip install numpy matplotlib scipy
```

**Or install individually:**

```bash
pip install numpy
pip install matplotlib
pip install scipy
```

**For Python 3 specifically (if needed):**

```bash
pip3 install numpy matplotlib scipy
```

### 2.3 Verify Installation

To check if libraries are installed correctly, run:

```bash
python -c "import numpy, matplotlib, scipy; print('✅ All libraries installed successfully!')"
```

---

## 📁 Step 3: Prepare Your Workshop Files

### 3.1 Required Files

Make sure you have all these files in the same folder:

1. ✅ `workshop_step1_load_real_ecg.py` - Load ECG data
2. ✅ `workshop_step2_filter_real_ecg.py` - Filter noise
3. ✅ `workshop_step3_detect_real_peaks.py` - Detect R peaks
4. ✅ `workshop_step4_calculate_hr_fixed.py` - Calculate heart rate
5. ✅ `workshop_synthetic_ecg_all_in_one.py` - Complete synthetic ECG demo
6. ✅ `ecg.mat` - Your ECG data file (if working with real data)

### 3.2 File Organisation

```
📂 ECG_Workshop/
├── 📄 workshop_step1_load_real_ecg.py
├── 📄 workshop_step2_filter_real_ecg.py
├── 📄 workshop_step3_detect_real_peaks.py
├── 📄 workshop_step4_calculate_hr_fixed.py
├── 📄 workshop_synthetic_ecg_all_in_one.py
├── 📄 ecg.mat (your ECG data)
└── 📄 README.md (this file)
```

---

## 🚀 Step 4: Run the Workshop (Step-by-Step)

### Option A: Working with REAL ECG Data

Follow these steps **in order** if you have an `ecg.mat` file:

#### **Step 1: Load Real ECG Signal**

```bash
python workshop_step1_load_real_ecg.py
```

**What this does:**
- Loads your ECG data from `ecg.mat`
- Extracts 60 seconds of ECG signal
- Visualises the signal in different time windows
- Creates `workshop_data.npz` for the next step

**Expected output:**
- Console output showing signal statistics
- Three plots displaying ECG signal
- File created: `workshop_data.npz`

---

#### **Step 2: Filter Real ECG Signal**

```bash
python workshop_step2_filter_real_ecg.py
```

**What this does:**
- Adds noise to the ECG signal (for demonstration)
- Applies a bandpass filter (0.5-40 Hz)
- Removes noise and baseline wander
- Compares original, noisy, and filtered signals

**Expected output:**
- Signal quality analysis
- Comparison plots
- Updated `workshop_data.npz`

---

#### **Step 3: Detect R Peaks**

```bash
python workshop_step3_detect_real_peaks.py
```

**What this does:**
- Detects R peaks (heartbeats) in the filtered ECG
- Uses adaptive threshold method
- Validates detection quality
- Calculates RR intervals

**Expected output:**
- Number of detected peaks
- Heart rate range
- Visualisation with marked R peaks
- Updated `workshop_data.npz`

---

#### **Step 4: Calculate Heart Rate**

```bash
python workshop_step4_calculate_hr_fixed.py
```

**What this does:**
- Calculates instantaneous heart rate
- Analyses heart rate variability (HRV)
- Generates comprehensive visualisations
- Exports results to CSV and text files

**Expected output:**
- Detailed heart rate statistics
- HRV metrics (SDNN, RMSSD, pNN50)
- Comprehensive analysis plots
- Files created:
  - `ecg_heart_rate_results.csv`
  - `ecg_summary_statistics.txt`
  - `ecg_complete_analysis.png`

---

### Option B: Working with SYNTHETIC ECG Data

If you **don't have real ECG data**, run the all-in-one synthetic ECG script:

```bash
python workshop_synthetic_ecg_all_in_one.py
```

**What this does:**
- Generates realistic synthetic ECG signal
- Performs all 4 steps automatically
- Demonstrates complete processing pipeline
- No `ecg.mat` file required!

**Expected output:**
- Complete analysis from signal generation to heart rate calculation
- Comprehensive visualisation
- Files created:
  - `synthetic_ecg_results.csv`
  - `synthetic_ecg_summary.txt`
  - `synthetic_ecg_workshop_complete.png`

---

## 📊 Understanding Your Results

### Heart Rate Metrics

- **Average Heart Rate:** Normal range is 60-100 BPM
  - < 60 BPM = Bradycardia (slow heart rate)
  - > 100 BPM = Tachycardia (fast heart rate)

### Heart Rate Variability (HRV) Metrics

- **SDNN (Standard Deviation of NN intervals):**
  - < 50 ms = Low HRV (may indicate stress)
  - 50-100 ms = Normal HRV
  - > 100 ms = High HRV (good cardiac health)

- **RMSSD (Root Mean Square of Successive Differences):**
  - Indicates parasympathetic nervous system activity
  - Higher values = better cardiac health

- **pNN50 (Percentage of NN intervals > 50ms difference):**
  - Measures high-frequency heart rate variability
  - Higher values = greater variability (healthier)

---

## 🔍 Troubleshooting Common Issues

### Issue 1: "ecg.mat file not found"

**Solution:**
- Make sure `ecg.mat` is in the same folder as the Python scripts
- Or use the synthetic ECG option (all-in-one script)

### Issue 2: "workshop_data.npz not found"

**Solution:**
- Run the scripts **in order** starting from Step 1
- Each step depends on the previous step's output

### Issue 3: Import errors (numpy, matplotlib, scipy)

**Solution:**
```bash
pip install --upgrade numpy matplotlib scipy
```

### Issue 4: Plots not showing

**Solution:**
- Make sure you're not running in a headless environment
- If using Jupyter, add: `%matplotlib inline`
- If using IDE, check if plot windows are blocked

### Issue 5: "Not enough peaks detected"

**Solution:**
- Your ECG signal quality might be poor
- Try adjusting the threshold in Step 3 (line 45)
- Check if filtering parameters need adjustment in Step 2

---

## 📈 What You Should See

### Step 1 Output Example:
```
Loading real ECG signal from .mat file...
✓ Found variable: 'ecg'
✓ Successfully loaded ECG data
  Total samples: 15360
  Total duration: 60.0 seconds
  Sampling frequency: 256 Hz
```

### Step 4 Final Output Example:
```
HEART RATE ANALYSIS
==================================================
Average Heart Rate: 75.3 BPM
Heart Rate Variability: 62 ms (SDNN)
Total Heartbeats Analyzed: 75
==================================================
```

---

## 🎯 Learning Objectives Checklist

After completing this workshop, you should be able to:

- [ ] Load and visualise ECG data from files
- [ ] Understand sampling frequency and signal duration
- [ ] Apply digital filters to remove noise
- [ ] Implement peak detection algorithms
- [ ] Calculate heart rate from ECG signals
- [ ] Analyse heart rate variability (HRV)
- [ ] Generate professional analysis reports
- [ ] Interpret clinical ECG metrics

---

## 📚 Additional Resources

### Learn More About ECG:
- ECG waveforms: P wave, QRS complex, T wave
- Clinical significance of HRV
- Arrhythmia detection

### Improve Your Skills:
1. Try different filtering parameters in Step 2
2. Experiment with peak detection thresholds in Step 3
3. Analyse ECG signals with different heart rates
4. Compare synthetic vs. real ECG characteristics

---

## 💡 Tips for Success

1. **Run scripts in order** - Each step builds on the previous one
2. **Read the console output** - It provides important information
3. **Examine the plots carefully** - Visual inspection is crucial
4. **Check generated files** - CSV files can be opened in Excel
5. **Experiment with parameters** - Try changing values to see effects
6. **Ask questions** - Contact us if you need help!

---

## 📧 Support & Contact

**Need help or have questions?**

- **Email:** torabisignals@gmail.com
- **Company:** Torabi Signals LTD, United Kingdom

For licensing inquiries or custom workshops, please contact us.

---

## 📝 License & Copyright

Copyright © 2024 Torabi Signals LTD. All rights reserved.

This educational material is licensed and proprietary to Torabi Signals LTD.

---

## 🎉 Ready to Start?

1. ✅ Check Python version
2. ✅ Install required libraries
3. ✅ Organise your files
4. ✅ Run Step 1!

**Have a great learning experience! 🚀**

---

## 📊 Quick Reference Commands

```bash
# Check Python version
python --version

# Install libraries
pip install numpy matplotlib scipy

# Run complete workflow (REAL ECG)
python workshop_step1_load_real_ecg.py
python workshop_step2_filter_real_ecg.py
python workshop_step3_detect_real_peaks.py
python workshop_step4_calculate_hr_fixed.py

# Run synthetic ECG demo (NO real data needed)
python workshop_synthetic_ecg_all_in_one.py
```

---

**Last Updated:** November 2024  
**Version:** 1.0  
**Workshop Duration:** 60-90 minutes

**Good luck with your ECG signal processing journey! 📈❤️**
