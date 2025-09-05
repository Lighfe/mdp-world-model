#!/usr/bin/env python3
# test_subprocess.py - Simple test to verify subprocess approach works

import subprocess
import sys
import os
import time
from pathlib import Path

def test_single_subprocess():
    """Test if we can run a single train_drm.py subprocess successfully"""
    
    # Use your base config
    config_path = "configs/base.yaml"
    
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    cmd = [
        sys.executable,
        "neural_networks/train_drm.py", 
        config_path,
        "--multi_run"
    ]
    
    print(f"Testing single subprocess...")
    print(f"Command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    
    start_time = time.time()
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,  # 2 minute timeout
            cwd=str(Path(__file__).parent)
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: Subprocess completed in {runtime:.1f}s")
            print("Last few lines of output:")
            for line in result.stdout.split('\n')[-5:]:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"✗ FAILED: Exit code {result.returncode} after {runtime:.1f}s")
            print("STDERR:")
            print(result.stderr[-500:])  # Last 500 chars
            print("STDOUT:")
            print(result.stdout[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: Process hung for >120s")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("SUBPROCESS TEST")
    print("="*50)
    
    success = test_single_subprocess()
    
    print("="*50)
    if success:
        print("✓ Subprocess approach works! Use the full multi-run script.")
    else:
        print("✗ Subprocess approach has issues. Check errors above.")
    print("="*50)