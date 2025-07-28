# LoopyCut

> 🌀 A Python-based CLI tool to analyze screen recordings, detect visually seamless loops, and trim videos accordingly.

LoopyCut intelligently analyzes video frames to detect perfect loop points and creates seamless looped videos automatically.

## Features

- ✅ **Intelligent Loop Detection**: Uses advanced frame comparison algorithms (SSIM, histogram analysis, perceptual hashing)
- ✅ **Flexible Parameters**: Control loop duration, similarity thresholds, and analysis windows
- ✅ **Multiple Comparison Methods**: Choose from SSIM, histogram, hash, or combined analysis
- ✅ **Resolution Control**: Resize output with crop, pad, or center strategies
- ✅ **Speed Adjustment**: Change playback speed while maintaining loop quality
- ✅ **Audio Handling**: Include or exclude audio as needed
- ✅ **Progress Feedback**: Real-time progress bars and detailed logging
- ✅ **Metadata Export**: Save loop information for future reference

## Installation

### Prerequisites

- Python 3.11 or higher
- FFmpeg (for video processing)

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

# Analyze specific time range
python loopycut.py input.mp4 output.mp4 --start 10 --stop 30
```

### Advanced Examples

```bash
# High precision with custom resolution
python loopycut.py input.mp4 output.mp4 \
  --similarity 99 \
  --resolution 1920x1080 \
  --resize-strategy crop

# Speed up and exclude audio
python loopycut.py input.mp4 output.mp4 \
  --speed 1.5 \
  --no-audio

# Custom buffers and method
python loopycut.py input.mp4 output.mp4 \
  --buffer-start 0.5 \
  --buffer-stop 1.0 \
  --method combined \
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
| `--similarity` | Match threshold (0-100) | `98` |
| `--start` | Start time for analysis (seconds) | `0.0` |
| `--stop` | Stop time for analysis (seconds) | `end` |
| `--buffer` | Equal buffer before/after loop | `0.0` |
| `--buffer-start` | Buffer before loop start | `0.0` |
| `--buffer-stop` | Buffer after loop end | `0.0` |
| `--resolution` | Output resolution (e.g. "1920x1080") | `original` |
| `--speed` | Playback speed multiplier | `1.0` |
| `--resize-strategy` | Resolution handling: crop/pad/center | `center` |
| `--method` | Comparison method: ssim/histogram/hash/combined | `combined` |
| `--audio/--no-audio` | Include audio in output | `True` |
| `--verbose` | Enable detailed output | `False` |

### Frame-based Control

```bash
# Use frame numbers instead of time
python loopycut.py input.mp4 output.mp4 \
  --start-frame 300 \
  --stop-frame 900
```

### Comparison Methods

- **`combined`** (default): Weighted combination of SSIM and histogram analysis
- **`ssim`**: Structural Similarity Index - best for detecting structural changes
- **`histogram`**: Color histogram comparison - good for color-based analysis
- **`hash`**: Perceptual hashing - fastest, good for identical frames

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
python loopycut.py screen_recording.mp4 demo_loop.mp4 \
  --length auto \
  --similarity 95 \
  --no-audio
```

### Game Clips

Create engaging gameplay loops:

```bash
python loopycut.py gameplay.mp4 highlight_loop.mp4 \
  --method combined \
  --speed 1.2 \
  --resolution 1280x720
```

### Animation Sequences

Loop animated content seamlessly:

```bash
python loopycut.py animation.mp4 perfect_loop.mp4 \
  --similarity 99 \
  --method ssim \
  --buffer 0.1
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

- Use `--method hash` for fastest analysis
- Limit analysis to relevant time ranges
- Use lower resolution for initial testing
- Consider frame-based parameters for precision

## Development

### Project Structure

```
loopycut/
├── frame_analyzer.py    # Frame extraction and comparison
├── loop_detector.py     # Loop detection algorithms
├── video_trimmer.py     # Video processing and output
├── cli.py              # Command-line interface
├── utils.py            # Utility functions
├── loopycut.py         # Main entry point
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Running Tests

```bash
# Test with sample video
python loopycut.py sample_video.mp4 test_output.mp4 --verbose

# Check system info
python loopycut.py --info

# Analyze video without processing
python cli.py info sample_video.mp4
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Roadmap

Phase 1 (Current):
- ✅ Core loop detection and trimming
- ✅ Command-line interface
- ✅ Multiple comparison methods
- ✅ Progress feedback and logging

Phase 2 (Planned):
- Export to GIF format
- Batch processing support
- JSON metadata export
- DaVinci Resolve XML/EDL export

Phase 3 (Future):
- Electron-based GUI
- DaVinci-style interface
- Real-time preview
- Advanced editing features
