# LoopyCut

> 🌀 A Python-based CLI tool to analyze screen recordings, detect visually seamless loops, and trim videos accordingly.

LoopyCut intelligently analyzes video frames to detect perfect loop points and creates seamless looped videos automatically.

## Features

- ✅ **Intelligent Loop Detection**: Uses advanced frame comparison algorithms (SSIM, histogram analysis, perceptual hashing)
- ✅ **GPU Acceleration**: Apple Silicon M1/M2 optimized with Numba JIT compilation (100-700x speedup)
- ✅ **Smart Frame Processing**: Automatic size reduction and downsampling with perfect time accuracy
- ✅ **Multiple Analysis Methods**: Fast hash, batch SSIM, hybrid, and traditional combined analysis
- ✅ **Flexible Parameters**: Control loop duration, similarity thresholds, and analysis windows
- ✅ **Resolution Control**: Resize output with crop, pad, or center strategies
- ✅ **Speed Adjustment**: Change playback speed while maintaining loop quality
- ✅ **Audio Handling**: Include or exclude audio as needed
- ✅ **Progress Feedback**: Real-time progress bars and detailed logging
- ✅ **Metadata Export**: Save loop information for future reference

## Installation

### Prerequisites

- Python 3.11 or higher
- FFmpeg (for video processing)
- Recommended: Apple Silicon Mac (M1/M2) for optimal GPU acceleration

### Install FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html) or use [Chocolatey](https://chocolatey.org/):
```bash
choco install ffmpeg
```

### Install LoopyCut

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/loopycut-app.git
cd loopycut-app
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python loopycut.py --info
```

## Quick Start

### Basic Usage

```bash
# Automatic loop detection
python loopycut.py input.mp4 output.mp4

# Specify desired loop length
python loopycut.py input.mp4 output.mp4 --length 5

# Analyze specific time range (flexible time formats)
python loopycut.py input.mp4 output.mp4 --start 00:00:14 --stop 00:00:32
python loopycut.py input.mp4 output.mp4 --start 1:30 --stop 2:45
python loopycut.py input.mp4 output.mp4 --start 14.5 --stop 32
```

### Advanced Examples

```bash
# High precision with custom resolution
python loopycut.py input.mp4 output.mp4 \
  --similarity 99 \
  --resolution 1920x1080 \
  --resize-strategy crop

# GPU-accelerated ultra-fast processing
python loopycut.py input.mp4 output.mp4 \
  --method fast_hash \
  --downsample 4 \
  --gpu

# Hybrid method for best speed + accuracy
python loopycut.py input.mp4 output.mp4 \
  --method hybrid \
  --similarity 95 \
  --gpu

# Speed up and exclude audio
python loopycut.py input.mp4 output.mp4 \
  --speed 1.5 \
  --no-audio

# Create 11-second output with 3x speed (uses 33s of source)
python loopycut.py input.mp4 output.mp4 \
  --output-length 11 \
  --speed 3.0

# Custom buffers and CPU-only mode
python loopycut.py input.mp4 output.mp4 \
  --buffer-start 0.5 \
  --buffer-stop 1.0 \
  --method combined \
  --no-gpu \
  --verbose
```

## Command Reference

### Required Arguments

- `INPUT`: Path to input video file
- `OUTPUT`: Path for output video file

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--length` | Loop length in seconds or "auto" | `auto` |
| `--output-length` | Desired final output duration (auto calculates source length based on speed) | `None` |
| `--similarity` | Match threshold (0-100) | `98` |
| `--start` | Start time for analysis (supports HH:MM:SS, MM:SS, or seconds) | `0.0` |
| `--stop` | Stop time for analysis (supports HH:MM:SS, MM:SS, or seconds) | `end` |
| `--buffer` | Equal buffer before/after loop | `0.0` |
| `--buffer-start` | Buffer before loop start | `0.0` |
| `--buffer-stop` | Buffer after loop end | `0.0` |
| `--resolution` | Output resolution (e.g. "1920x1080") | `original` |
| `--speed` | Playback speed multiplier | `1.0` |
| `--resize-strategy` | Resolution handling: crop/pad/center | `center` |
| `--method` | Comparison method: ssim/histogram/hash/combined/fast_hash/batch_ssim/hybrid | `combined` |
| `--gpu/--no-gpu` | Enable/disable GPU acceleration | `True` |
| `--downsample` | Extract every Nth frame (1=all frames) | `1` |
| `--audio/--no-audio` | Include audio in output | `True` |
| `--verbose` | Enable detailed output | `False` |

### Frame-based Control

```bash
# Use frame numbers instead of time
python loopycut.py input.mp4 output.mp4 \
  --start-frame 300 \
  --stop-frame 900
```

### Time Format Support

The `--start` and `--stop` parameters support multiple time formats for flexibility:

```bash
# HH:MM:SS format
python loopycut.py input.mp4 output.mp4 --start 00:01:30 --stop 00:02:15

# MM:SS format  
python loopycut.py input.mp4 output.mp4 --start 1:30 --stop 2:15

# Seconds (decimal supported)
python loopycut.py input.mp4 output.mp4 --start 90 --stop 135.5
```

### Comparison Methods

**Traditional Methods:**
- **`combined`** (default): Weighted combination of SSIM and histogram analysis
- **`ssim`**: Structural Similarity Index - best for detecting structural changes
- **`histogram`**: Color histogram comparison - good for color-based analysis
- **`hash`**: Perceptual hashing - fastest, good for identical frames

**GPU-Accelerated Methods:**
- **`fast_hash`**: Fast perceptual hashing (100-700x speedup)
- **`batch_ssim`**: GPU-optimized SSIM calculations
- **`hybrid`**: Hash pre-filtering + SSIM verification (best balance of speed + accuracy)

### Resize Strategies

- **`center`** (default): Scale to fit and center with black bars if needed
- **`crop`**: Scale and crop to fill target resolution exactly
- **`pad`**: Scale to fit within target and pad with black bars

## Video Information

Get detailed information about any video:

```bash
python cli.py info input.mp4
```

## Examples by Use Case

### Screen Recordings

Perfect for UI demonstrations and app walkthroughs:

```bash
# Ultra-fast processing for screen recordings
python loopycut.py screen_recording.mp4 demo_loop.mp4 \
  --method fast_hash \
  --downsample 2 \
  --similarity 95 \
  --no-audio

# High quality for detailed UI work
python loopycut.py screen_recording.mp4 demo_loop.mp4 \
  --method hybrid \
  --similarity 98 \
  --no-audio
```

### Game Clips

Create engaging gameplay loops:

```bash
# Fast processing for gameplay highlights
python loopycut.py gameplay.mp4 highlight_loop.mp4 \
  --method fast_hash \
  --speed 1.2 \
  --resolution 1280x720 \
  --downsample 4

# Quality processing for cinematic clips
python loopycut.py gameplay.mp4 cinematic_loop.mp4 \
  --method hybrid \
  --similarity 99 \
  --resolution 1920x1080
```

### Animation Sequences

Loop animated content seamlessly:

```bash
# Perfect loops for animations
python loopycut.py animation.mp4 perfect_loop.mp4 \
  --method hybrid \
  --similarity 99 \
  --buffer 0.1

# Quick preview loops
python loopycut.py animation.mp4 preview_loop.mp4 \
  --method fast_hash \
  --downsample 2 \
  --similarity 90
```

## Output

LoopyCut generates:

1. **Looped video file** at the specified output path
2. **Metadata JSON file** (optional) containing loop information and settings
3. **Console output** with analysis results and loop statistics

### Sample Output

```
LoopyCut - Video Loop Creation Tool
========================================
Input: screen_recording.mp4
Output: demo_loop.mp4
Desired length: auto
Similarity threshold: 98%

Getting video information...
Video duration: 45.3s
Resolution: 1920x1080
Frame rate: 30.00 fps

Analyzing video for loop opportunities...
Extracting frames: 100%|████████████| 1359/1359 [00:12<00:00, 112.18frames/s]
Comparing frames (combined): 100%|████████████| 923041/923041 [01:23<00:00, 11087.23pairs/s]

Found 3 loop candidate(s):
----------------------------------------------------------------------
Loop 1:
  Time: 12.34s - 17.89s
  Duration: 5.55s (167 frames)
  Quality: 0.987
  Similarity: 0.991
  Final Score: 0.989

Selected loop:
  Time: 0:12.34 - 0:17.89
  Duration: 5.6s
  Quality score: 0.987

Creating looped video...
Processing video with FFmpeg...
Final duration: 5.6s
File size: 2.3 MB

✓ Successfully created looped video: demo_loop.mp4
```

## Troubleshooting

### Common Issues

**No loops found:**
- Lower similarity threshold: `--similarity 90`
- Try different comparison method: `--method histogram`
- Expand analysis window: `--start 0 --stop 60`

**Poor loop quality:**
- Increase similarity threshold: `--similarity 99`
- Use SSIM method: `--method ssim`
- Add small buffers: `--buffer 0.1`

**FFmpeg errors:**
- Ensure FFmpeg is installed and in PATH
- Check input file format compatibility
- Verify sufficient disk space

**Memory issues with large videos:**
- Limit analysis window: `--start X --stop Y`
- Use frame-based analysis for precision

### Performance Tips

**GPU Acceleration (Apple Silicon M1/M2):**
- Use `--method fast_hash` for maximum speed (100-700x faster)
- Use `--method hybrid` for best balance of speed and accuracy
- Use `--downsample 2-8` for large videos to extract fewer frames
- GPU acceleration is enabled by default, use `--no-gpu` to disable

**Memory Optimization:**
- Frame analysis automatically reduced to 480p (configurable)
- Use `--downsample` for additional memory savings
- Limit analysis to relevant time ranges: `--start X --stop Y`

**Speed vs Quality Trade-offs:**
```bash
# Maximum speed (up to 700x faster)
python loopycut.py large_video.mp4 output.mp4 --method fast_hash --downsample 8

# Balanced speed + accuracy  
python loopycut.py video.mp4 output.mp4 --method hybrid --downsample 2

# Maximum quality (slower)
python loopycut.py video.mp4 output.mp4 --method combined --similarity 99 --no-gpu
```

**Traditional Performance Tips:**
- Use `--method hash` for fastest analysis
- Limit analysis window: `--start X --stop Y`
- Use lower resolution for initial testing
- Consider frame-based parameters for precision

## Development

### Project Structure

```
loopycut/
├── frame_analyzer.py        # Frame extraction and comparison
├── frame_analyzer_gpu.py    # GPU-accelerated frame analysis
├── loop_detector.py         # Loop detection algorithms
├── video_trimmer.py         # Video processing and output
├── cli.py                   # Command-line interface
├── utils.py                 # Utility functions
├── loopycut.py              # Main entry point
├── requirements.txt         # Dependencies
├── test_gpu_performance.py  # Performance testing
├── demo_gpu.py              # GPU acceleration demo
└── README.md                # This file
```

### Running Tests

```bash
# Test with sample video
python loopycut.py sample_video.mp4 test_output.mp4 --verbose

# Check system info and GPU capabilities
python loopycut.py --info

# Test GPU performance
python test_gpu_performance.py

# GPU acceleration demo
python demo_gpu.py

# Analyze video without processing
python cli.py info sample_video.mp4
```

## License

[LoopyCut](https://github.com/carmelosantana/loopycut-cli) © 2025 by [Carmelo Santana](https://carmelosantana.com) is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
