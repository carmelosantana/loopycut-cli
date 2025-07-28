"""
LoopyCut - Frame Analysis Module

This module handles the extraction and analysis of video frames to detect
visual similarities for loop detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import hashlib
from tqdm import tqdm


class FrameAnalyzer:
    """
    Analyzes video frames to detect visual similarities for loop detection.
    
    Uses multiple comparison methods including structural similarity (SSIM),
    histogram comparison, and perceptual hashing to find matching frames.
    """
    
    def __init__(self, similarity_threshold: float = 0.98):
        """
        Initialize the frame analyzer.
        
        Args:
            similarity_threshold: Threshold for frame similarity (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.frames_cache: List[np.ndarray] = []
        self.frame_hashes: List[str] = []
        
    def extract_frames(self, video_path: str, start_frame: int = 0, 
                      end_frame: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from video within the specified range.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame number (0-indexed)
            end_frame: Ending frame number (None for end of video)
            
        Returns:
            List of extracted frames as numpy arrays
            
        Raises:
            ValueError: If video cannot be opened or frame range is invalid
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if end_frame is None:
            end_frame = total_frames
        
        # Validate frame range
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"Invalid start_frame: {start_frame}")
        if end_frame > total_frames:
            end_frame = total_frames
        if start_frame >= end_frame:
            raise ValueError("start_frame must be less than end_frame")
        
        frames = []
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames with progress bar
        with tqdm(total=end_frame - start_frame, 
                 desc="Extracting frames", 
                 unit="frames") as pbar:
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB for consistency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                pbar.update(1)
        
        cap.release()
        self.frames_cache = frames
        return frames
    
    def calculate_frame_hash(self, frame: np.ndarray) -> str:
        """
        Calculate perceptual hash of a frame for quick comparison.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Hexadecimal hash string
        """
        # Resize frame to small size for faster hashing
        small_frame = cv2.resize(frame, (8, 8))
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate average pixel value
        avg = gray.mean()
        
        # Create binary hash based on pixel values vs average
        hash_bits = []
        for pixel in gray.flatten():
            hash_bits.append('1' if pixel > avg else '0')
        
        # Convert binary string to hex
        hash_str = ''.join(hash_bits)
        return hex(int(hash_str, 2))[2:]
    
    def calculate_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            SSIM value between 0.0 and 1.0
        """
        # Convert to grayscale for SSIM calculation
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Ensure both frames have the same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Calculate SSIM using OpenCV
        # Note: OpenCV doesn't have SSIM, so we use a simplified version
        # Based on mean, variance, and covariance
        
        mu1 = gray1.mean()
        mu2 = gray2.mean()
        
        var1 = gray1.var()
        var2 = gray2.var()
        
        cov = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM formula constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        
        ssim = numerator / denominator
        return max(0.0, min(1.0, ssim))  # Clamp between 0 and 1
    
    def calculate_histogram_similarity(self, frame1: np.ndarray, 
                                     frame2: np.ndarray) -> float:
        """
        Calculate histogram similarity between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Similarity value between 0.0 and 1.0
        """
        # Calculate histograms for each color channel
        hist1 = []
        hist2 = []
        
        for channel in range(3):  # RGB channels
            h1 = cv2.calcHist([frame1], [channel], None, [256], [0, 256])
            h2 = cv2.calcHist([frame2], [channel], None, [256], [0, 256])
            hist1.append(h1)
            hist2.append(h2)
        
        # Calculate correlation for each channel
        correlations = []
        for h1, h2 in zip(hist1, hist2):
            correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            correlations.append(correlation)
        
        # Return average correlation across channels
        return np.mean(correlations)
    
    def find_similar_frames(self, frames: List[np.ndarray] = None, 
                           method: str = "combined") -> List[Tuple[int, int, float]]:
        """
        Find pairs of similar frames in the frame list.
        
        Args:
            frames: List of frames to analyze (uses cached frames if None)
            method: Comparison method ("ssim", "histogram", "hash", "combined")
            
        Returns:
            List of tuples (frame1_idx, frame2_idx, similarity_score)
        """
        if frames is None:
            frames = self.frames_cache
        
        if not frames:
            raise ValueError("No frames available for analysis")
        
        similar_pairs = []
        total_comparisons = len(frames) * (len(frames) - 1) // 2
        
        with tqdm(total=total_comparisons, 
                 desc=f"Comparing frames ({method})", 
                 unit="pairs") as pbar:
            
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    similarity = self._calculate_similarity(
                        frames[i], frames[j], method
                    )
                    
                    if similarity >= self.similarity_threshold:
                        similar_pairs.append((i, j, similarity))
                    
                    pbar.update(1)
        
        # Sort by similarity score (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def _calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray, 
                            method: str) -> float:
        """
        Calculate similarity between two frames using specified method.
        
        Args:
            frame1: First frame
            frame2: Second frame
            method: Comparison method
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if method == "ssim":
            return self.calculate_ssim(frame1, frame2)
        elif method == "histogram":
            return self.calculate_histogram_similarity(frame1, frame2)
        elif method == "hash":
            hash1 = self.calculate_frame_hash(frame1)
            hash2 = self.calculate_frame_hash(frame2)
            # Simple hash comparison (exact match)
            return 1.0 if hash1 == hash2 else 0.0
        elif method == "combined":
            # Weighted combination of methods
            ssim_score = self.calculate_ssim(frame1, frame2)
            hist_score = self.calculate_histogram_similarity(frame1, frame2)
            
            # Weight: 60% SSIM, 40% histogram
            return 0.6 * ssim_score + 0.4 * hist_score
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get basic information about the video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'fourcc': cap.get(cv2.CAP_PROP_FOURCC)
        }
        
        cap.release()
        return info
