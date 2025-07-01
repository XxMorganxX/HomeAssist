# 🔊 AEC Parameter Tuning Guide

This guide helps you fine-tune your Acoustic Echo Cancellation (AEC) parameters for optimal audio filtering performance.

## 🎯 Quick Start

### 1. Test Current Settings
```bash
python test_aec_performance.py
# Select option 1 for a quick 10-second test
```

### 2. Interactive Parameter Tuning
```bash
python core/aec_tuner.py
```

### 3. Compare Multiple Configurations
```bash
python test_aec_performance.py
# Select option 2 to compare different optimized settings
```

## 📊 Understanding AEC Parameters

### **AEC_FILTER_LENGTH** (200-500)
- **What it does**: Number of filter taps in the NLMS adaptive filter
- **Higher values (400-500)**:
  - ✅ Better echo removal for complex environments
  - ✅ Handles longer echo patterns
  - ❌ More CPU usage
  - ❌ Slightly higher latency
- **Lower values (200-300)**:
  - ✅ Faster processing
  - ✅ Lower latency
  - ❌ May miss complex echo patterns

### **AEC_STEP_SIZE** (0.01-0.1)
- **What it does**: Controls how quickly the filter adapts to changes
- **Higher values (0.06-0.1)**:
  - ✅ Faster adaptation to changing audio
  - ✅ Better for dynamic environments
  - ❌ Risk of instability/oscillation
  - ❌ May overcorrect
- **Lower values (0.01-0.03)**:
  - ✅ More stable operation
  - ✅ Less likely to cause artifacts
  - ❌ Slower to adapt to changes

### **AEC_DELAY_SAMPLES**
- **What it does**: Compensates for the acoustic delay between speaker and microphone
- **How to estimate**:
  - Physical delay: `distance_meters / 343 * sample_rate`
  - System delay: Add 50-200ms for audio processing
  - Example: 1 meter distance = ~47 samples + ~1600 samples system = ~1650 total

### **AEC_REFERENCE_BUFFER_SEC** (3.0-10.0)
- **What it does**: How long to keep reference audio for echo cancellation
- **Longer buffers (8-10 seconds)**:
  - ✅ Handles longer echo tails
  - ✅ Better for reverberant rooms
  - ❌ More memory usage
- **Shorter buffers (3-5 seconds)**:
  - ✅ Less memory usage
  - ✅ Faster processing
  - ❌ May miss delayed echoes

## 🎛️ Pre-configured Settings

Your `config.py` now includes several optimized profiles:

### **Current Default** (Improved Balance)
```python
AEC_FILTER_LENGTH = 400      # Increased for better echo removal
AEC_STEP_SIZE = 0.03         # Reduced for stability
AEC_DELAY_SAMPLES = 800      # Keep current
AEC_REFERENCE_BUFFER_SEC = 8.0  # Increased for longer echo tails
```

### **For Aggressive Echo Cancellation**
```python
AEC_FILTER_LENGTH = 500
AEC_STEP_SIZE = 0.02
AEC_DELAY_SAMPLES = 1000
AEC_REFERENCE_BUFFER_SEC = 10.0
```

### **For Fast Adaptation**
```python
AEC_FILTER_LENGTH = 250
AEC_STEP_SIZE = 0.08
AEC_DELAY_SAMPLES = 600
AEC_REFERENCE_BUFFER_SEC = 4.0
```

### **For Stable Operation**
```python
AEC_FILTER_LENGTH = 350
AEC_STEP_SIZE = 0.02
AEC_DELAY_SAMPLES = 800
AEC_REFERENCE_BUFFER_SEC = 6.0
```

## 🔍 Troubleshooting Common Issues

### 🔊 **Persistent Echo**
**Symptoms**: You hear your assistant's voice echoing back
**Solutions**:
1. Increase `AEC_FILTER_LENGTH` to 500
2. Decrease `AEC_STEP_SIZE` to 0.02
3. Increase `AEC_REFERENCE_BUFFER_SEC` to 10.0
4. Check speaker-microphone distance and adjust `AEC_DELAY_SAMPLES`

### 📢 **Audio Feedback/Howling**
**Symptoms**: High-pitched squealing or feedback
**Solutions**:
1. Reduce `AEC_STEP_SIZE` to 0.01 (very conservative)
2. Set `AEC_FILTER_LENGTH` to 400
3. Reduce `AEC_DELAY_SAMPLES` if speaker and mic are close
4. Check physical setup - increase distance between speaker and mic

### 🎵 **Audio Distortion**
**Symptoms**: Processed audio sounds warped or artificial
**Solutions**:
1. Reduce `AEC_FILTER_LENGTH` to 250
2. Set `AEC_STEP_SIZE` to 0.03
3. Reduce `AEC_REFERENCE_BUFFER_SEC` to 5.0
4. Consider disabling AEC temporarily to test

### 🔇 **Background Noise Issues**
**Symptoms**: AEC overreacts to background noise
**Solutions**:
1. Use conservative settings: `AEC_STEP_SIZE = 0.02`
2. Set `AEC_FILTER_LENGTH` to 350
3. Increase `AEC_REFERENCE_BUFFER_SEC = 8.0`
4. Improve physical environment (reduce noise sources)

### ⏱️ **Audio Delay/Latency**
**Symptoms**: Noticeable delay in audio processing
**Solutions**:
1. Reduce `AEC_FILTER_LENGTH` to 200-250
2. Increase `AEC_STEP_SIZE` to 0.08 (faster adaptation)
3. Reduce `AEC_REFERENCE_BUFFER_SEC` to 3.0
4. Optimize `AEC_DELAY_SAMPLES` for your setup

## 🧪 Testing Workflow

### **Step 1: Baseline Test**
```bash
python test_aec_performance.py
# Option 1: Quick test with current settings
```

### **Step 2: Identify Issues**
Listen for:
- Echo (repeating voice)
- Feedback (squealing)
- Distortion (unnatural sound)
- Delay (lag in processing)

### **Step 3: Apply Targeted Settings**
```bash
python core/aec_tuner.py
# Option 2: Generate config for specific issue
```

### **Step 4: Test and Compare**
```bash
python test_aec_performance.py
# Option 2: Compare multiple configurations
```

### **Step 5: Fine-tune**
Adjust parameters based on test results:
- If echo persists: Increase filter length, decrease step size
- If unstable: Decrease step size, moderate filter length
- If delayed: Reduce filter length, increase step size

## 📈 Performance Monitoring

### **Real-time Statistics**
Watch for these metrics during testing:
- **Echo Reduction**: Should be > 10 dB for good performance
- **Frames Processed**: Confirms AEC is active
- **Reference Buffer Usage**: Should match your buffer setting

### **Expected Performance**
- **Excellent**: > 15 dB echo reduction, stable operation
- **Good**: 10-15 dB echo reduction, minimal artifacts
- **Fair**: 5-10 dB echo reduction, some residual echo
- **Poor**: < 5 dB echo reduction, needs tuning

## 🔧 Advanced Tips

### **Physical Setup Optimization**
1. **Speaker Placement**: Away from microphone (> 1 meter if possible)
2. **Room Acoustics**: Soft furnishings reduce echo
3. **Microphone Quality**: Better microphones = better AEC performance
4. **Volume Levels**: Moderate speaker volume reduces echo strength

### **Delay Estimation**
Use the interactive tuner to estimate optimal delay:
```bash
python core/aec_tuner.py
# Option 1: Estimate delay parameters
```

### **Performance Logging**
The system automatically logs performance data to `aec_tuning_log.json` for analysis.

### **Environment-Specific Tuning**
- **Small Room**: Lower filter length, shorter buffer
- **Large Room**: Higher filter length, longer buffer
- **Noisy Environment**: Conservative step size, longer buffer
- **Quiet Environment**: Can use more aggressive settings

## 🎯 Quick Commands Reference

```bash
# Quick test current settings
python test_aec_performance.py

# Interactive parameter tuning
python core/aec_tuner.py

# Compare optimized configurations
python test_aec_performance.py  # Option 2

# Test specific issue type
python test_aec_performance.py  # Option 4
```

## 🆘 Need Help?

1. **Start with the improved default settings** (already applied to your config)
2. **Run a quick test** to establish baseline performance
3. **Use the interactive tuner** for guided parameter selection
4. **Test iteratively** - small changes often work better than large ones
5. **Monitor the echo reduction metric** as your primary performance indicator

Remember: AEC tuning is iterative. Start with the provided profiles and fine-tune based on your specific environment and audio setup! 