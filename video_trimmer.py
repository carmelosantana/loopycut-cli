"""
LoopyCut - Video Trimming Module

This module handles the video trimming and processing operations using FFmpeg
to create the final looped video output with specified parameters.
"""

import ffmpeg
import os
from typing import Optional, Tuple
from pathlib import Path


class VideoTrimmer:
    """
    Handles video trimming and processing operations to create looped videos.
    
    Uses FFmpeg through the ffmpeg-python library to perform precise video
    trimming, resolution adjustments, speed changes, and audio processing.
    """
    
    def __init__(self):
        """Initialize the video trimmer."""
        self.last_operation_info = {}
    
    def trim_video(self, input_path: str, output_path: str,
                  start_time: float, end_time: float,
                  buffer_start: float = 0.0, buffer_end: float = 0.0,
                  resolution: Optional[str] = None,
                  speed: float = 1.0, include_audio: bool = True,
                  resize_strategy: str = "center") -> bool:
        """
        Trim video to create a loop segment with specified parameters.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            start_time: Start time in seconds
            end_time: End time in seconds
            buffer_start: Additional seconds before start_time
            buffer_end: Additional seconds after end_time
            resolution: Target resolution (e.g., "1920x1080", None for original)
            speed: Playback speed multiplier (1.0 = normal speed)
            include_audio: Whether to include audio in output
            resize_strategy: How to handle resolution changes ("crop", "pad", "center")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate actual trim points with buffers
            actual_start = max(0, start_time - buffer_start)
            actual_end = end_time + buffer_end
            duration = actual_end - actual_start
            
            print(f"Trimming video from {actual_start:.2f}s to {actual_end:.2f}s "
                  f"(duration: {duration:.2f}s)")
            
            # Check if input file exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video not found: {input_path}")
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Start building FFmpeg pipeline
            input_stream = ffmpeg.input(input_path, ss=actual_start, t=duration)
            
            # Get video and audio streams
            video_stream = input_stream.video
            audio_stream = input_stream.audio if include_audio else None
            
            # Apply speed adjustment if needed
            if speed != 1.0:
                video_stream = video_stream.filter('setpts', f'{1/speed}*PTS')
                if audio_stream:
                    audio_stream = audio_stream.filter('atempo', speed)
            
            # Apply resolution changes if specified
            if resolution:
                video_stream = self._apply_resolution_change(
                    video_stream, resolution, resize_strategy
                )
            
            # Build output arguments
            output_args = {
                'vcodec': 'libx264',  # Use H.264 codec
                'preset': 'medium',   # Encoding speed vs quality
                'crf': 18             # High quality (lower = better quality)
            }
            
            # Add audio settings if including audio
            if include_audio and audio_stream:
                output_args.update({
                    'acodec': 'aac',
                    'audio_bitrate': '128k'
                })
                
                # Combine video and audio
                output_stream = ffmpeg.output(
                    video_stream, audio_stream, output_path, **output_args
                )
            else:
                # Video only
                output_stream = ffmpeg.output(
                    video_stream, output_path, **output_args
                )
            
            # Run the FFmpeg command
            print("Processing video with FFmpeg...")
            ffmpeg.run(output_stream, quiet=False, overwrite_output=True)
            
            # Store operation info for reference
            self.last_operation_info = {
                'input_path': input_path,
                'output_path': output_path,
                'start_time': actual_start,
                'end_time': actual_end,
                'duration': duration,
                'resolution': resolution,
                'speed': speed,
                'include_audio': include_audio,
                'resize_strategy': resize_strategy
            }
            
            print(f"Successfully created trimmed video: {output_path}")
            return True
            
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e}")
            if e.stderr:
                print(f"FFmpeg stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"Error trimming video: {e}")
            return False
    
    def _apply_resolution_change(self, video_stream, target_resolution: str,
                                strategy: str):
        """
        Apply resolution changes to video stream using specified strategy.
        
        Args:
            video_stream: FFmpeg video stream
            target_resolution: Target resolution (e.g., "1920x1080")
            strategy: Resize strategy ("crop", "pad", "center")
            
        Returns:
            Modified video stream
        """
        try:
            # Parse target resolution
            width, height = map(int, target_resolution.split('x'))
        except ValueError:
            raise ValueError(f"Invalid resolution format: {target_resolution}")
        
        if strategy == "crop":
            # Scale and crop to fill target resolution
            video_stream = video_stream.filter('scale', width, height, force_original_aspect_ratio='increase')
            video_stream = video_stream.filter('crop', width, height)
        
        elif strategy == "pad":
            # Scale and pad to fit target resolution
            video_stream = video_stream.filter('scale', width, height, force_original_aspect_ratio='decrease')
            video_stream = video_stream.filter('pad', width, height, x='(ow-iw)/2', y='(oh-ih)/2', color='black')
        
        elif strategy == "center":
            # Scale to fit and center (may have black bars)
            video_stream = video_stream.filter('scale', width, height, force_original_aspect_ratio='decrease')
            video_stream = video_stream.filter('pad', width, height, x='(ow-iw)/2', y='(oh-ih)/2', color='black')
        
        else:
            raise ValueError(f"Unknown resize strategy: {strategy}")
        
        return video_stream
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about a video file using FFmpeg probe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video information
        """
        try:
            probe = ffmpeg.probe(video_path)
            
            # Find video stream
            video_stream = next(
                stream for stream in probe['streams'] 
                if stream['codec_type'] == 'video'
            )
            
            # Find audio stream (if exists)
            audio_streams = [
                stream for stream in probe['streams'] 
                if stream['codec_type'] == 'audio'
            ]
            
            # Extract information
            info = {
                'duration': float(probe['format']['duration']),
                'size_bytes': int(probe['format']['size']),
                'video': {
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
                    'codec': video_stream['codec_name'],
                    'pixel_format': video_stream.get('pix_fmt', 'unknown')
                },
                'audio': {
                    'streams': len(audio_streams),
                    'codec': audio_streams[0]['codec_name'] if audio_streams else None,
                    'sample_rate': int(audio_streams[0]['sample_rate']) if audio_streams else None,
                    'channels': int(audio_streams[0]['channels']) if audio_streams else None
                } if audio_streams else None
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
    
    def create_test_loop(self, input_path: str, output_path: str,
                        loop_duration: float = 5.0, num_repetitions: int = 3) -> bool:
        """
        Create a test loop by repeating a segment multiple times.
        
        Args:
            input_path: Path to input video
            output_path: Path for test output
            loop_duration: Duration of each loop in seconds
            num_repetitions: Number of times to repeat the loop
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a temporary loop file
            temp_loop = output_path.replace('.mp4', '_temp_loop.mp4')
            
            # First, create the base loop
            success = self.trim_video(
                input_path, temp_loop,
                start_time=0, end_time=loop_duration,
                include_audio=False  # Skip audio for test
            )
            
            if not success:
                return False
            
            # Create input list for concatenation
            inputs = []
            for i in range(num_repetitions):
                inputs.append(ffmpeg.input(temp_loop))
            
            # Concatenate the loops
            concat_stream = ffmpeg.concat(*inputs, v=1, a=0)
            
            # Output the final test video
            output_stream = ffmpeg.output(
                concat_stream, output_path,
                vcodec='libx264', crf=18, y=True
            )
            
            print(f"Creating test loop with {num_repetitions} repetitions...")
            ffmpeg.run(output_stream, quiet=True, overwrite_output=True)
            
            # Clean up temporary file
            if os.path.exists(temp_loop):
                os.remove(temp_loop)
            
            print(f"Test loop created: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating test loop: {e}")
            return False
    
    def validate_output(self, output_path: str) -> bool:
        """
        Validate that the output file was created successfully.
        
        Args:
            output_path: Path to output file
            
        Returns:
            True if file exists and is valid, False otherwise
        """
        if not os.path.exists(output_path):
            print(f"Output file not found: {output_path}")
            return False
        
        try:
            # Try to get video info to validate file
            info = self.get_video_info(output_path)
            if not info:
                print(f"Output file appears to be invalid: {output_path}")
                return False
            
            print(f"Output validation successful:")
            print(f"  Duration: {info['duration']:.2f}s")
            print(f"  Resolution: {info['video']['width']}x{info['video']['height']}")
            print(f"  Size: {info['size_bytes'] / (1024*1024):.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error validating output: {e}")
            return False
