#!/usr/bin/env python3
# debug_python_paths.py

import sys
import os
import subprocess
import shutil

print("="*50)
print("PYTHON ENVIRONMENT DEBUG")
print("="*50)

# Check main process Python
print(f"Main process Python executable: {sys.executable}")
print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'NOT SET')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'NOT SET')}")
print(f"PATH first entry: {os.environ.get('PATH', '').split(':')[0]}")

# Check which python
python_which = shutil.which('python')
print(f"which python: {python_which}")

python3_which = shutil.which('python3')
print(f"which python3: {python3_which}")

# Test torch import in main process
try:
    import torch
    print(f"✓ Main process can import torch: {torch.__version__}")
except ImportError as e:
    print(f"✗ Main process cannot import torch: {e}")

print("\n" + "="*50)
print("SUBPROCESS PYTHON CHECK")
print("="*50)

# Test subprocess with sys.executable
print(f"Testing subprocess with sys.executable: {sys.executable}")
try:
    result = subprocess.run([
        sys.executable, 
        "-c", 
        "import sys; print(f'Subprocess Python: {sys.executable}'); import torch; print(f'Torch: {torch.__version__}')"
    ], capture_output=True, text=True, timeout=10)
    
    if result.returncode == 0:
        print("✓ Subprocess with sys.executable works:")
        print(f"  STDOUT: {result.stdout.strip()}")
    else:
        print("✗ Subprocess with sys.executable failed:")
        print(f"  STDERR: {result.stderr.strip()}")
        print(f"  STDOUT: {result.stdout.strip()}")
        
except Exception as e:
    print(f"✗ Subprocess test failed: {e}")

# Test subprocess with conda python path
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    conda_python = f"{conda_prefix}/bin/python"
    print(f"\nTesting subprocess with conda python: {conda_python}")
    
    try:
        result = subprocess.run([
            conda_python, 
            "-c", 
            "import sys; print(f'Conda subprocess Python: {sys.executable}'); import torch; print(f'Torch: {torch.__version__}')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Subprocess with conda python works:")
            print(f"  STDOUT: {result.stdout.strip()}")
        else:
            print("✗ Subprocess with conda python failed:")
            print(f"  STDERR: {result.stderr.strip()}")
            print(f"  STDOUT: {result.stdout.strip()}")
            
    except Exception as e:
        print(f"✗ Conda subprocess test failed: {e}")

print("\n" + "="*50)
print("RECOMMENDATION")
print("="*50)

if conda_prefix and os.path.exists(f"{conda_prefix}/bin/python"):
    print(f"✓ Use explicit conda Python path: {conda_prefix}/bin/python")
else:
    print("? Check conda environment setup")