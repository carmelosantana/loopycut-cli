"""
LoopyCut - Loop Detection Module

This module implements algorithms to detect seamless loops in video sequences
by analyzing frame similarities and finding optimal loop points.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
from frame_analyzer import FrameAnalyzer
# Import for type hinting, but avoid circular import
try:
    from frame_analyzer_gpu import GPUFrameAnalyzer
except ImportError:
    GPUFrameAnalyzer = None


class LoopDetector:
    """
    Detects optimal loop points in video sequences based on frame similarity analysis.
    
    This class uses the FrameAnalyzer to find frames that are visually similar
    and determines the best loop boundaries based on user-specified criteria.
    """
    
    def __init__(self, frame_analyzer):
        """
        Initialize the loop detector.
        
        Args:
            frame_analyzer: Instance of FrameAnalyzer or GPUFrameAnalyzer for frame comparison
        """
        self.frame_analyzer = frame_analyzer
        self.detected_loops: List[dict] = []
    
    def detect_loops(self, video_path: str, 
                    desired_length: Union[str, float] = "auto",
                    start_time: float = 0.0,
                    end_time: Optional[float] = None,
                    start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None,
                    downsample_factor: int = 1,
                    method: str = "combined") -> List[dict]:
        """
        Detect possible loops in the video within the specified range.
        
        Args:
            video_path: Path to the input video file
            desired_length: Desired loop length in seconds ("auto" for best match)
            start_time: Start time in seconds for analysis window
            end_time: End time in seconds for analysis window (None for end)
            start_frame: Start frame number (overrides start_time if provided)
            end_frame: End frame number (overrides end_time if provided)
            downsample_factor: Extract every Nth frame for faster processing
            method: Frame comparison method to use
            
        Returns:
            List of detected loop candidates with metadata
        """
        # Get video information
        video_info = self.frame_analyzer.get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        # Convert time boundaries to frame numbers if needed
        if start_frame is None:
            start_frame = int(start_time * fps)
        if end_frame is None:
            if end_time is not None:
                end_frame = int(end_time * fps)
            else:
                end_frame = total_frames
        
        # Validate frame boundaries
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        print(f"Analyzing frames {start_frame} to {end_frame} "
              f"({(end_frame - start_frame) / fps:.2f} seconds)")
        
        # Extract frames in the specified range
        if hasattr(self.frame_analyzer, 'extract_frames') and hasattr(self.frame_analyzer, 'find_similar_frames_optimized'):
            # GPU Frame Analyzer
            frames = self.frame_analyzer.extract_frames(
                video_path, start_frame, end_frame, downsample_factor
            )
            # Use GPU-optimized similarity detection
            similar_pairs = self.frame_analyzer.find_similar_frames_optimized(frames, method)
        else:
            # Regular Frame Analyzer
            frames = self.frame_analyzer.extract_frames(
                video_path, start_frame, end_frame
            )
            # Use regular similarity detection
            similar_pairs = self.frame_analyzer.find_similar_frames(frames, method)
        
        if not similar_pairs:
            print("No similar frames found. Try adjusting similarity threshold.")
            return []
        
        # Convert similar pairs to loop candidates
        loop_candidates = self._analyze_loop_candidates(
            similar_pairs, frames, fps, desired_length, start_frame, downsample_factor
        )
        
        # Rank and filter loop candidates
        ranked_loops = self._rank_loop_candidates(loop_candidates, desired_length)
        
        self.detected_loops = ranked_loops
        return ranked_loops
    
    def _analyze_loop_candidates(self, similar_pairs: List[Tuple[int, int, float]],
                                frames: List[np.ndarray], fps: float,
                                desired_length: Union[str, float],
                                frame_offset: int, downsample_factor: int = 1) -> List[dict]:
        """
        Analyze similar frame pairs to create loop candidates.
        
        Args:
            similar_pairs: List of (start_idx, end_idx, similarity) tuples
            frames: List of extracted frames
            fps: Video frame rate
            desired_length: Desired loop length
            frame_offset: Offset of first frame in the full video
            downsample_factor: Factor used for downsampling (for time correction)
            
        Returns:
            List of loop candidate dictionaries
        """
        candidates = []
        
        with tqdm(similar_pairs, desc="Analyzing loop candidates", unit="pairs") as pbar:
            for start_idx, end_idx, similarity in pbar:
                
                # Convert extracted frame indices to original video frame indices
                if hasattr(self.frame_analyzer, 'get_original_frame_index'):
                    # GPU analyzer with proper frame index mapping
                    original_start_frame = self.frame_analyzer.get_original_frame_index(start_idx)
                    original_end_frame = self.frame_analyzer.get_original_frame_index(end_idx)
                else:
                    # Original analyzer - account for downsampling manually
                    original_start_frame = frame_offset + (start_idx * downsample_factor)
                    original_end_frame = frame_offset + (end_idx * downsample_factor)
                
                # Calculate loop properties using original frame indices
                loop_frames = original_end_frame - original_start_frame
                loop_duration = loop_frames / fps
                
                # Skip very short loops (less than 0.5 seconds)
                if loop_duration < 0.5:
                    continue
                
                # Skip if desired length is specified and loop is too different
                if (isinstance(desired_length, (int, float)) and 
                    abs(loop_duration - desired_length) > desired_length * 0.2):
                    continue
                
                # Calculate quality metrics using extracted frame indices
                quality_score = self._calculate_loop_quality(
                    frames, start_idx, end_idx, similarity
                )
                
                candidate = {
                    'start_frame': original_start_frame,
                    'end_frame': original_end_frame,
                    'start_time': original_start_frame / fps,
                    'end_time': original_end_frame / fps,
                    'duration': loop_duration,
                    'frame_count': loop_frames,
                    'similarity_score': similarity,
                    'quality_score': quality_score,
                    'fps': fps,
                    'downsample_factor': downsample_factor  # Store for reference
                }
                
                candidates.append(candidate)
        
        return candidates
    
    def _calculate_loop_quality(self, frames: List[np.ndarray], 
                               start_idx: int, end_idx: int, 
                               similarity: float) -> float:
        """
        Calculate a quality score for a potential loop.
        
        Args:
            frames: List of frames
            start_idx: Start frame index
            end_idx: End frame index
            similarity: Similarity score between start and end frames
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from similarity
        quality = similarity
        
        # Analyze loop consistency by checking intermediate frames
        loop_frames = frames[start_idx:end_idx + 1]
        if len(loop_frames) > 10:  # Only analyze if loop has enough frames
            # Sample a few frames throughout the loop
            sample_indices = np.linspace(0, len(loop_frames) - 1, 
                                       min(5, len(loop_frames)), dtype=int)
            
            # Check for consistent motion/changes throughout the loop
            motion_consistency = self._analyze_motion_consistency(
                [loop_frames[i] for i in sample_indices]
            )
            
            # Weight: 70% similarity, 30% motion consistency
            quality = 0.7 * similarity + 0.3 * motion_consistency
        
        return quality
    
    def _analyze_motion_consistency(self, sample_frames: List[np.ndarray]) -> float:
        """
        Analyze motion consistency across sample frames in a loop.
        
        Args:
            sample_frames: Sample frames from the loop
            
        Returns:
            Motion consistency score between 0.0 and 1.0
        """
        if len(sample_frames) < 2:
            return 1.0
        
        # Calculate frame differences
        differences = []
        for i in range(len(sample_frames) - 1):
            diff = cv2.absdiff(
                cv2.cvtColor(sample_frames[i], cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(sample_frames[i + 1], cv2.COLOR_RGB2GRAY)
            )
            differences.append(np.mean(diff))
        
        # Check if differences are relatively consistent
        if not differences:
            return 1.0
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # Lower standard deviation relative to mean indicates consistency
        if mean_diff == 0:
            return 1.0
        
        consistency = max(0.0, 1.0 - (std_diff / mean_diff))
        return consistency
    
    def _rank_loop_candidates(self, candidates: List[dict], 
                             desired_length: Union[str, float]) -> List[dict]:
        """
        Rank loop candidates based on quality and user preferences.
        
        Args:
            candidates: List of loop candidate dictionaries
            desired_length: Desired loop length
            
        Returns:
            Sorted list of loop candidates (best first)
        """
        if not candidates:
            return []
        
        # Calculate ranking scores
        for candidate in candidates:
            score = candidate['quality_score']
            
            # Bonus for matching desired length
            if isinstance(desired_length, (int, float)):
                length_diff = abs(candidate['duration'] - desired_length)
                length_penalty = length_diff / desired_length
                score *= (1.0 - min(0.5, length_penalty))  # Max 50% penalty
            
            # Bonus for longer loops (more content)
            duration_bonus = min(0.1, candidate['duration'] / 10.0)  # Max 10% bonus
            score += duration_bonus
            
            candidate['final_score'] = score
        
        # Sort by final score (highest first)
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates
    
    def get_best_loop(self, video_path: str, **kwargs) -> Optional[dict]:
        """
        Get the single best loop candidate.
        
        Args:
            video_path: Path to the input video file
            **kwargs: Arguments passed to detect_loops()
            
        Returns:
            Best loop candidate dictionary or None if no loops found
        """
        loops = self.detect_loops(video_path, **kwargs)
        return loops[0] if loops else None
    
    def print_loop_summary(self, loops: List[dict]) -> None:
        """
        Print a summary of detected loops.
        
        Args:
            loops: List of loop dictionaries
        """
        if not loops:
            print("No loops detected.")
            return
        
        print(f"\nFound {len(loops)} loop candidate(s):")
        print("-" * 70)
        
        for i, loop in enumerate(loops[:5]):  # Show top 5
            print(f"Loop {i + 1}:")
            print(f"  Time: {loop['start_time']:.2f}s - {loop['end_time']:.2f}s")
            print(f"  Duration: {loop['duration']:.2f}s ({loop['frame_count']} frames)")
            print(f"  Quality: {loop['quality_score']:.3f}")
            print(f"  Similarity: {loop['similarity_score']:.3f}")
            print(f"  Final Score: {loop['final_score']:.3f}")
            print()
        
        if len(loops) > 5:
            print(f"... and {len(loops) - 5} more candidates")


# Import cv2 here since it's used in the motion consistency analysis
import cv2
