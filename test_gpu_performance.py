#!/usr/bin/env python3
"""
LoopyCut GPU Performance Test

This script tests the performance difference between the original CPU implementation
and the new GPU-accelerated implementation.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from frame_analyzer import FrameAnalyzer
from frame_analyzer_gpu import GPUFrameAnalyzer


def create_test_frames(n_frames: int = 100, width: int = 320, height: int = 240) -> list:
    """Create synthetic test frames for performance testing."""
    frames = []
    
    print(f"Creating {n_frames} synthetic test frames ({width}x{height})...")
    
    for i in range(n_frames):
        # Create frames with some similarity patterns
        if i % 10 == 0:  # Every 10th frame is similar to frame 0
            base_frame = np.random.randint(0, 128, (height, width, 3), dtype=np.uint8)
        else:
            base_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some noise
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    return frames


def benchmark_analyzer(analyzer, frames, method_name, analysis_method="combined"):
    """Benchmark a frame analyzer."""
    print(f"\n{'='*50}")
    print(f"Testing {method_name}")
    print(f"{'='*50}")
    
    # Test frame analysis
    start_time = time.time()
    
    if hasattr(analyzer, 'find_similar_frames_optimized'):
        similar_pairs = analyzer.find_similar_frames_optimized(frames, method=analysis_method)
    else:
        similar_pairs = analyzer.find_similar_frames(frames, method="combined")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Analysis completed in: {duration:.2f} seconds")
    print(f"Found {len(similar_pairs)} similar frame pairs")
    
    if similar_pairs:
        print(f"Top 5 matches:")
        for i, (frame1, frame2, score) in enumerate(similar_pairs[:5]):
            print(f"  {i+1}. Frames {frame1}-{frame2}: {score:.4f}")
    
    return duration, len(similar_pairs)


def main():
    """Run performance comparison tests."""
    print("LoopyCut GPU Performance Test")
    print("=" * 60)
    
    # Test with different frame counts to show scaling
    test_configs = [
        (50, "small test"),
        (100, "medium test"),
        # (200, "large test"),  # Commented out for faster testing
    ]
    
    results = {}
    
    for n_frames, test_name in test_configs:
        print(f"\n🎬 Running {test_name} with {n_frames} frames")
        print("-" * 40)
        
        # Create test frames
        test_frames = create_test_frames(n_frames)
        
        # Test original CPU implementation
        print("\n1️⃣ Original CPU Implementation")
        cpu_analyzer = FrameAnalyzer(similarity_threshold=0.85)
        cpu_time, cpu_matches = benchmark_analyzer(
            cpu_analyzer, test_frames, "CPU (Original)", "combined"
        )
        
        # Test GPU-accelerated implementation - Hash method
        print("\n2️⃣ GPU Implementation - Fast Hash")
        gpu_analyzer = GPUFrameAnalyzer(similarity_threshold=0.85)
        gpu_hash_time, gpu_hash_matches = benchmark_analyzer(
            gpu_analyzer, test_frames, "GPU (Fast Hash)", "fast_hash"
        )
        
        # Test GPU-accelerated implementation - Hybrid method
        print("\n3️⃣ GPU Implementation - Hybrid")
        gpu_hybrid_time, gpu_hybrid_matches = benchmark_analyzer(
            gpu_analyzer, test_frames, "GPU (Hybrid)", "hybrid"
        )
        
        # Calculate speedup
        hash_speedup = cpu_time / gpu_hash_time if gpu_hash_time > 0 else float('inf')
        hybrid_speedup = cpu_time / gpu_hybrid_time if gpu_hybrid_time > 0 else float('inf')
        
        results[test_name] = {
            'cpu_time': cpu_time,
            'gpu_hash_time': gpu_hash_time,
            'gpu_hybrid_time': gpu_hybrid_time,
            'hash_speedup': hash_speedup,
            'hybrid_speedup': hybrid_speedup,
            'cpu_matches': cpu_matches,
            'gpu_hash_matches': gpu_hash_matches,
            'gpu_hybrid_matches': gpu_hybrid_matches,
        }
        
        print(f"\n📊 Performance Summary for {test_name}:")
        print(f"  CPU Time:           {cpu_time:.2f}s")
        print(f"  GPU Hash Time:      {gpu_hash_time:.2f}s  (🚀 {hash_speedup:.1f}x speedup)")
        print(f"  GPU Hybrid Time:    {gpu_hybrid_time:.2f}s  (🚀 {hybrid_speedup:.1f}x speedup)")
        print(f"  Matches Found:      CPU={cpu_matches}, Hash={gpu_hash_matches}, Hybrid={gpu_hybrid_matches}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎯 FINAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Hash Method:   {result['hash_speedup']:.1f}x faster than CPU")
        print(f"  Hybrid Method: {result['hybrid_speedup']:.1f}x faster than CPU")
    
    # Show GPU capabilities
    gpu_analyzer = GPUFrameAnalyzer()
    stats = gpu_analyzer.get_performance_stats()
    
    print(f"\n🔧 SYSTEM CAPABILITIES:")
    print(f"  GPU Acceleration:   {'✅' if stats['gpu_acceleration'] else '❌'}")
    print(f"  OpenCV GPU:         {'✅' if stats['opencv_gpu'] else '❌'}")
    print(f"  CPU Threads:        {stats['cpu_threads']}")
    print(f"  Numba JIT:          {'✅' if stats['numba_available'] else '❌'}")
    
    print(f"\n🎉 Test completed! The GPU-accelerated version should be significantly faster.")
    print(f"💡 For even better performance on large videos, consider:")
    print(f"   - Using downsample_factor > 1 when extracting frames")
    print(f"   - Using 'fast_hash' method for initial screening")
    print(f"   - Using 'hybrid' method for better accuracy")


if __name__ == "__main__":
    main()
