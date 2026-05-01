#!/usr/bin/env python3
"""
CUDA Architecture Detection and Configuration for exo-cuda

Detects GPU compute capability and sets appropriate CUDA architecture flags.
Fixes cudaErrorInvalidDeviceFunction on GTX 1650 (sm_75) and similar devices.

Usage:
    python3 cuda_arch_setup.py          # Detect and display
    python3 cuda_arch_setup.py --set    # Set environment variables
"""

import os
import sys
import subprocess
import json
from typing import Dict, List, Optional, Tuple


# CUDA compute capability to architecture mapping
COMPUTE_CAP_TO_ARCH = {
    (5, 0): 'sm_50',  # Maxwell
    (5, 2): 'sm_52',  # Maxwell (Mobile)
    (5, 3): 'sm_53',  # Maxwell (Tegra)
    (6, 0): 'sm_60',  # Pascal (Tesla P100)
    (6, 1): 'sm_61',  # Pascal (GTX 1080, GTX 1070, GTX 1060)
    (6, 2): 'sm_62',  # Pascal (Tegra X1)
    (7, 0): 'sm_70',  # Volta (V100)
    (7, 2): 'sm_72',  # Volta (DGX-2)
    (7, 5): 'sm_75',  # Turing (GTX 1650, GTX 1660, RTX 2080)
    (8, 0): 'sm_80',  # Ampere (A100)
    (8, 6): 'sm_86',  # Ampere (RTX 3080, RTX 3090, RTX 3060)
    (8, 7): 'sm_87',  # Ampere (Jetson AGX Orin)
    (8, 9): 'sm_89',  # Ada Lovelace (RTX 4090)
    (9, 0): 'sm_90',  # Hopper (H100)
}


def detect_cuda_gpus() -> List[Dict]:
    """Detect NVIDIA GPUs and their compute capabilities using nvidia-smi."""
    gpus = []
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap,driver_version,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name = parts[0]
                    compute_cap = float(parts[1])
                    driver_version = parts[2]
                    memory_mb = int(parts[3])
                    
                    # Convert compute capability (e.g., 7.5) to (7, 5)
                    major = int(compute_cap)
                    minor = int((compute_cap - major) * 10)
                    
                    gpus.append({
                        'name': name,
                        'compute_capability': compute_cap,
                        'major': major,
                        'minor': minor,
                        'driver_version': driver_version,
                        'memory_mb': memory_mb,
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        print(f"Warning: Could not detect GPUs via nvidia-smi: {e}")
    
    return gpus


def get_arch_string(compute_cap: Tuple[int, int]) -> str:
    """Get CUDA architecture string for compute capability."""
    return COMPUTE_CAP_TO_ARCH.get(compute_cap, f'sm_{compute_cap[0]}{compute_cap[1]}')


def generate_cudaarchs(gpus: List[Dict]) -> str:
    """Generate CUDAARCHS string for all detected GPUs."""
    if not gpus:
        # Default to common architectures
        return '70;75;80;86'
    
    archs = set()
    for gpu in gpus:
        cap = (gpu['major'], gpu['minor'])
        if cap in COMPUTE_CAP_TO_ARCH:
            arch_str = COMPUTE_CAP_TO_ARCH[cap]
            # Extract numeric part (e.g., 'sm_75' -> '75')
            arch_num = arch_str.split('_')[1]
            archs.add(arch_num)
    
    return ';'.join(sorted(archs)) if archs else '70;75;80;86'


def display_gpu_info(gpus: List[Dict]) -> None:
    """Display GPU information in a formatted table."""
    if not gpus:
        print("\n⚠️  No NVIDIA GPUs detected")
        print("   Using default CUDA architectures: 70;75;80;86\n")
        return
    
    print(f"\n{'='*60}")
    print(f"  CUDA Architecture Detection (exo-cuda)")
    print(f"{'='*60}\n")
    
    for i, gpu in enumerate(gpus):
        cap = (gpu['major'], gpu['minor'])
        arch = get_arch_string(cap)
        
        print(f"  GPU {i}: {gpu['name']}")
        print(f"    Compute Capability: {gpu['compute_capability']}")
        print(f"    Architecture: {arch}")
        print(f"    Driver: {gpu['driver_version']}")
        print(f"    Memory: {gpu['memory_mb']} MB")
        print()
    
    # Generate CUDAARCHS
    cudaarchs = generate_cudaarchs(gpus)
    print(f"  Recommended CUDAARCHS: {cudaarchs}")
    print()
    
    # Check for common issues
    issues = []
    for gpu in gpus:
        cap = (gpu['major'], gpu['minor'])
        if cap not in COMPUTE_CAP_TO_ARCH:
            issues.append(f"  ⚠️  {gpu['name']} (sm_{cap[0]}{cap[1]}) not in known architectures list")
    
    if issues:
        print("  Issues:")
        for issue in issues:
            print(issue)
        print()


def set_environment_variables(gpus: List[Dict]) -> None:
    """Set CUDA environment variables for build."""
    cudaarchs = generate_cudaarchs(gpus)
    
    print(f"\n  Setting environment variables:")
    print(f"    export CUDAARCHS={cudaarchs}")
    print(f"    export TORCH_CUDA_ARCH_LIST='{cudaarchs.replace(';', ';')}'")
    print()
    
    # Set in current process
    os.environ['CUDAARCHS'] = cudaarchs
    os.environ['TORCH_CUDA_ARCH_LIST'] = cudaarchs.replace(';', ';')
    
    print("  ✅ Environment variables set for current session")
    print()
    print("  To make permanent, add to ~/.bashrc or ~/.zshrc:")
    print(f"    export CUDAARCHS={cudaarchs}")
    print(f"    export TORCH_CUDA_ARCH_LIST='{cudaarchs.replace(';', ';')}'")
    print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect CUDA GPU architecture and set build flags'
    )
    parser.add_argument(
        '--set',
        action='store_true',
        help='Set CUDAARCHS environment variables'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    
    args = parser.parse_args()
    
    # Detect GPUs
    gpus = detect_cuda_gpus()
    
    if args.json:
        output = {
            'gpus': gpus,
            'cudaarchs': generate_cudaarchs(gpus),
        }
        print(json.dumps(output, indent=2))
    else:
        display_gpu_info(gpus)
        
        if args.set:
            set_environment_variables(gpus)


if __name__ == '__main__':
    main()
