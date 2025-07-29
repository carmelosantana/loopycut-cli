#!/usr/bin/env python3
"""
LoopyCut - Command Line Interface

A Python tool for creating seamless video loops through intelligent frame analysis.
Analyzes video frames to detect visual similarities and creates perfectly looped videos.
"""

import click
import json
import os
import sys
from typing import Optional

# Import our modules
from frame_analyzer import FrameAnalyzer
from frame_analyzer_gpu import GPUFrameAnalyzer
from loop_detector import LoopDetector
from video_trimmer import VideoTrimmer
from utils import (
    validate_video_file, ensure_output_directory, format_time,
    format_duration, parse_time_string, save_loop_metadata,
    print_system_info, check_dependencies
)


class TimeType(click.ParamType):
    """Custom Click parameter type for parsing time strings."""
    name = 'time'
    
    def convert(self, value, param, ctx):
        if value is None:
            return None
        try:
            return parse_time_string(value)
        except ValueError as e:
            self.fail(f'{value!r} is not a valid time format. {str(e)}', param, ctx)


@click.command()
@click.argument('input', type=click.Path(), required=False)
@click.argument('output', type=click.Path(), required=False)
@click.option('--length', default="auto", 
              help='Length of desired loop in seconds (or "auto" for best match)')
@click.option('--buffer', type=float, default=0.0,
              help='Apply equal buffer before and after loop (seconds)')
@click.option('--buffer-start', type=float, default=None,
              help='Extra seconds added before loop start (overrides --buffer)')
@click.option('--buffer-stop', type=float, default=None,
              help='Extra seconds added after loop end (overrides --buffer)')
@click.option('--similarity', type=int, default=98,
              help='Match threshold for similar frames (0-100)')
@click.option('--start', '--start-time', type=TimeType(), default=0.0,
              help='Starting time for loop detection (supports HH:MM:SS, MM:SS, or seconds)')
@click.option('--stop', '--stop-time', type=TimeType(), default=None,
              help='Stop time to limit detection window (supports HH:MM:SS, MM:SS, or seconds)')
@click.option('--start-frame', type=int, default=None,
              help='Start frame (overrides --start-time if both given)')
@click.option('--stop-frame', type=int, default=None,
              help='Stop frame (overrides --stop-time if both given)')
@click.option('--resolution', type=str, default=None,
              help='Output resolution (e.g. "1920x1080")')
@click.option('--audio/--no-audio', default=True,
              help='Include audio in the output')
@click.option('--speed', type=float, default=1.0,
              help='Playback speed multiplier (e.g. 1.0 = normal)')
@click.option('--output-length', type=float, default=None,
              help='Desired final output duration in seconds (automatically calculates source length needed based on speed)')
@click.option('--resize-strategy', type=click.Choice(['crop', 'pad', 'center']),
              default='center', help='Strategy for resolution mismatch')
@click.option('--method', type=click.Choice(['ssim', 'histogram', 'hash', 'combined', 'fast_hash', 'batch_ssim', 'hybrid']),
              default='combined', help='Frame comparison method')
@click.option('--gpu/--no-gpu', default=True,
              help='Enable GPU acceleration (Apple Silicon/CUDA)')
@click.option('--downsample', type=int, default=1,
              help='Extract every Nth frame for faster processing (1=all frames)')
@click.option('--save-metadata/--no-save-metadata', default=True,
              help='Save loop metadata to JSON file')
@click.option('--loop-count', type=int, default=1,
              help='Number of different loops to create from the video (default: 1)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--info', is_flag=True,
              help='Show system information and exit')
def main(input: Optional[str], output: Optional[str], length: str, buffer: float,
         buffer_start: Optional[float], buffer_stop: Optional[float],
         similarity: int, start: float, stop: Optional[float],
         start_frame: Optional[int], stop_frame: Optional[int],
         resolution: Optional[str], audio: bool, speed: float, output_length: Optional[float],
         resize_strategy: str, method: str, gpu: bool, downsample: int,
         save_metadata: bool, loop_count: int, verbose: bool, info: bool):
    """
    LoopyCut - Create seamless video loops automatically.
    
    INPUT: Path to input video file
    OUTPUT: Path for output video file
    
    Examples:
    
        # Automatic loop detection
        loopycut input.mp4 output.mp4
        
        # 5-second loop with 99% similarity
        loopycut input.mp4 output.mp4 --length 5 --similarity 99
        
        # Analyze specific time range (supports multiple formats)
        loopycut input.mp4 output.mp4 --start 00:00:14 --stop 00:00:32
        loopycut input.mp4 output.mp4 --start 1:30 --stop 2:45
        loopycut input.mp4 output.mp4 --start 14.5 --stop 32
        
        # Custom resolution and speed
        loopycut input.mp4 output.mp4 --resolution 1280x720 --speed 1.5
        
        # Speed up 3x but keep 11 second output (uses 33s of source)
        loopycut input.mp4 output.mp4 --output-length 11 --speed 3.0
        
        # Create 3 different loops from the same video
        loopycut input.mp4 output.mp4 --loop-count 3 --similarity 85
    """
    
    # Show system info and exit if requested
    if info:
        print_system_info()
        return
    
    # If info flag is not set, validate required arguments
    if not input or not output:
        click.echo("Error: INPUT and OUTPUT arguments are required unless using --info", err=True)
        click.echo("Try 'loopycut.py --help' for help.", err=True)
        sys.exit(1)
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [dep for dep, available in deps.items() if not available]
    
    if missing_deps:
        click.echo(f"Error: Missing dependencies: {', '.join(missing_deps)}", err=True)
        click.echo("Install with: pip install -r requirements.txt", err=True)
        sys.exit(1)
    
    # Validate input file
    if not validate_video_file(input):
        click.echo(f"Error: Invalid video file: {input}", err=True)
        sys.exit(1)
    
    # Ensure output directory exists
    if not ensure_output_directory(output):
        click.echo(f"Error: Cannot create output directory for: {output}", err=True)
        sys.exit(1)
    
    # Parse length parameter
    if output_length is not None and length != "auto":
        # Check if user explicitly specified both length and output_length
        click.echo("Error: Cannot specify both --length and --output-length", err=True)
        sys.exit(1)
    
    if output_length is not None:
        # User specified desired output length - calculate source length needed
        if output_length <= 0:
            click.echo("Error: Output length must be positive", err=True)
            sys.exit(1)
        # Calculate source length needed: output_length * speed = source_length
        calculated_source_length = output_length * speed
        desired_length = calculated_source_length
        if verbose:
            click.echo(f"Output length requested: {output_length}s at {speed}x speed")
            click.echo(f"Source length needed: {calculated_source_length}s")
    elif length == "auto":
        desired_length = "auto"
    else:
        try:
            desired_length = float(length)
            if desired_length <= 0:
                raise ValueError("Length must be positive")
        except ValueError as e:
            click.echo(f"Error: Invalid length value: {length}", err=True)
            sys.exit(1)
    
    # Validate similarity threshold
    if not (0 <= similarity <= 100):
        click.echo("Error: Similarity must be between 0 and 100", err=True)
        sys.exit(1)

    # Validate loop count
    if loop_count < 1:
        click.echo("Error: Loop count must be at least 1", err=True)
        sys.exit(1)
    if loop_count > 10:
        click.echo("Error: Loop count cannot exceed 10 (performance limit)", err=True)
        sys.exit(1)

    # Convert similarity percentage to decimal
    similarity_threshold = similarity / 100.0
    
    # Handle buffer options
    if buffer_start is None:
        buffer_start = buffer
    if buffer_stop is None:
        buffer_stop = buffer
    
    if verbose:
        click.echo("LoopyCut - Video Loop Creation Tool")
        click.echo("=" * 40)
        click.echo(f"Input: {input}")
        click.echo(f"Output: {output}")
        click.echo(f"Desired length: {desired_length}")
        click.echo(f"Similarity threshold: {similarity}%")
        click.echo(f"Frame comparison method: {method}")
        click.echo(f"Audio: {'Yes' if audio else 'No'}")
        if resolution:
            click.echo(f"Resolution: {resolution} ({resize_strategy})")
        if speed != 1.0:
            click.echo(f"Speed: {speed}x")
        click.echo()
    
    try:
        # Initialize components
        if verbose:
            click.echo("Initializing video analysis...")
            click.echo(f"GPU acceleration: {'Enabled' if gpu else 'Disabled'}")
            if downsample > 1:
                click.echo(f"Downsampling: every {downsample} frames")
        
        # Choose analyzer based on GPU setting
        if gpu and method in ['fast_hash', 'batch_ssim', 'hybrid']:
            frame_analyzer = GPUFrameAnalyzer(similarity_threshold=similarity_threshold)
            if verbose:
                stats = frame_analyzer.get_performance_stats()
                click.echo(f"GPU capabilities: {stats}")
        else:
            frame_analyzer = FrameAnalyzer(similarity_threshold=similarity_threshold)
            if gpu and verbose:
                click.echo("Note: GPU methods require 'fast_hash', 'batch_ssim', or 'hybrid' method")
        
        loop_detector = LoopDetector(frame_analyzer)
        video_trimmer = VideoTrimmer()
        
        # Get video information
        if verbose:
            click.echo("Getting video information...")
        
        video_info = frame_analyzer.get_video_info(input)
        duration = video_info['duration_seconds']
        fps = video_info['fps']
        
        if verbose:
            click.echo(f"Video duration: {format_duration(duration)}")
            click.echo(f"Resolution: {video_info['width']}x{video_info['height']}")
            click.echo(f"Frame rate: {fps:.2f} fps")
            click.echo()
        
        # Detect loops
        click.echo("Analyzing video for loop opportunities...")
        
        loops = loop_detector.detect_loops(
            input,
            desired_length=desired_length,
            start_time=start,
            end_time=stop,
            start_frame=start_frame,
            end_frame=stop_frame,
            downsample_factor=downsample,
            method=method
        )
        
        if not loops:
            click.echo("No suitable loops found. Try adjusting parameters:", err=True)
            click.echo("  - Lower similarity threshold (--similarity)", err=True)
            click.echo("  - Different time range (--start, --stop)", err=True)
            click.echo("  - Different comparison method (--method)", err=True)
            sys.exit(1)
        
        # Show loop candidates
        if verbose or len(loops) > 1:
            loop_detector.print_loop_summary(loops)

        # Determine how many loops to create
        loops_to_create = min(loop_count, len(loops))
        
        if loops_to_create < loop_count:
            click.echo(f"Warning: Only found {loops_to_create} suitable loop(s), creating {loops_to_create} instead of {loop_count}")
        
        # Process each loop
        for i in range(loops_to_create):
            current_loop = loops[i]
            
            # Generate output filename
            if loop_count == 1:
                current_output = output
            else:
                # Add sequence number to filename
                base_name, ext = os.path.splitext(output)
                current_output = f"{base_name}_loop{i+1:02d}{ext}"
            
            click.echo(f"Creating loop {i+1}/{loops_to_create}:")
            click.echo(f"  Time: {format_time(current_loop['start_time'])} - {format_time(current_loop['end_time'])}")
            click.echo(f"  Duration: {format_duration(current_loop['duration'])}")
            click.echo(f"  Quality score: {current_loop['quality_score']:.3f}")
            click.echo(f"  Output: {current_output}")
            click.echo()

            # Create the looped video
            success = video_trimmer.trim_video(
                input_path=input,
                output_path=current_output,
                start_time=current_loop['start_time'],
                end_time=current_loop['end_time'],
                buffer_start=buffer_start,
                buffer_end=buffer_stop,
                resolution=resolution,
                speed=speed,
                include_audio=audio,
                resize_strategy=resize_strategy
            )

            if not success:
                click.echo(f"Error: Failed to create video output: {current_output}", err=True)
                continue

            # Validate output
            if not video_trimmer.validate_output(current_output):
                click.echo(f"Warning: Output validation failed for: {current_output}", err=True)

            # Save metadata if requested
            if save_metadata:
                metadata = {
                    'input_file': input,
                    'output_file': current_output,
                    'loop_info': current_loop,
                    'processing_options': {
                        'similarity_threshold': similarity,
                        'method': method,
                        'buffer_start': buffer_start,
                        'buffer_stop': buffer_stop,
                        'resolution': resolution,
                        'speed': speed,
                        'include_audio': audio,
                        'resize_strategy': resize_strategy,
                        'loop_count': loop_count,
                        'loop_index': i + 1
                    }
                }
                
                metadata_file = current_output.replace('.mp4', '.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

        # Summary message
        if loops_to_create == 1:
            click.echo(f"✓ Successfully created looped video: {output}")
        else:
            click.echo(f"✓ Successfully created {loops_to_create} looped videos:")
            for i in range(loops_to_create):
                if loop_count == 1:
                    current_output = output
                else:
                    base_name, ext = os.path.splitext(output)
                    current_output = f"{base_name}_loop{i+1:02d}{ext}"
                
                # Get file size
                try:
                    file_size = os.path.getsize(current_output) / (1024 * 1024)
                    click.echo(f"  {current_output} ({file_size:.1f} MB)")
                except:
                    click.echo(f"  {current_output}")

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)





@click.command()
@click.argument('video_path', type=click.Path(exists=True))
def info(video_path: str):
    """Show detailed information about a video file."""
    try:
        # Check dependencies
        deps = check_dependencies()
        if not all(deps.values()):
            click.echo("Error: Some dependencies are missing", err=True)
            return
        
        frame_analyzer = FrameAnalyzer()
        video_info = frame_analyzer.get_video_info(video_path)
        
        click.echo(f"Video Information: {video_path}")
        click.echo("=" * 50)
        click.echo(f"Duration: {format_duration(video_info['duration_seconds'])}")
        click.echo(f"Frames: {video_info['total_frames']}")
        click.echo(f"Frame Rate: {video_info['fps']:.2f} fps")
        click.echo(f"Resolution: {video_info['width']}x{video_info['height']}")
        
        # Try to get additional info using FFmpeg
        try:
            video_trimmer = VideoTrimmer()
            detailed_info = video_trimmer.get_video_info(video_path)
            
            if detailed_info:
                click.echo(f"File Size: {detailed_info['size_bytes'] / (1024*1024):.1f} MB")
                click.echo(f"Video Codec: {detailed_info['video']['codec']}")
                if detailed_info.get('audio'):
                    click.echo(f"Audio: {detailed_info['audio']['codec']} "
                              f"({detailed_info['audio']['channels']} channels)")
                else:
                    click.echo("Audio: None")
        except:
            pass  # Ignore FFmpeg errors for info command
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


# CLI group for multiple commands
@click.group()
def cli():
    """LoopyCut - Video Loop Creation Tool"""
    pass


# Add commands to the group
cli.add_command(main, name='create')
cli.add_command(info, name='info')


if __name__ == '__main__':
    # If called directly, run the main command
    main()
