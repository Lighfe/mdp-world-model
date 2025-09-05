# neural_networks/train_drm_subprocess_minimal.py
import os
import sys
import subprocess
import time
from pathlib import Path

def create_minimal_runner_script():
    """Create a minimal Python script that just imports and runs train_drm_model"""
    script_content = '''#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"[SUBPROCESS] Starting with config: {sys.argv[1]}")

try:
    print("[SUBPROCESS] About to import train_drm...")
    from neural_networks.train_drm import train_drm_model
    print("[SUBPROCESS] Import successful")
    
    print("[SUBPROCESS] About to call train_drm_model...")
    result = train_drm_model(sys.argv[1], multi_run=True)
    print("[SUBPROCESS] train_drm_model completed successfully")
    
except Exception as e:
    print(f"[SUBPROCESS] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[SUBPROCESS] Exiting successfully")
'''
    
    script_path = Path("neural_networks/minimal_runner.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def test_subprocess_variations():
    """Test different subprocess approaches to find what works"""
    
    config_path = "configs/base.yaml"
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    # Create minimal runner script
    runner_script = create_minimal_runner_script()
    
    print("="*60)
    print("TESTING SUBPROCESS VARIATIONS")
    print("="*60)
    
    # Test 1: Direct python call with minimal script
    print("\n1. Testing minimal runner script...")
    test_direct_call(runner_script, config_path)
    
    # Test 2: Using shell=True
    print("\n2. Testing with shell=True...")
    test_shell_call(runner_script, config_path)
    
    # Test 3: Staggered startup
    print("\n3. Testing staggered startup...")
    test_staggered_startup(runner_script, config_path)

def test_direct_call(runner_script, config_path):
    """Test direct subprocess call"""
    cmd = [sys.executable, str(runner_script), config_path]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,  # 1 minute timeout
            cwd=str(Path.cwd())
        )
        
        if result.returncode == 0:
            print("✓ SUCCESS: Direct call worked")
            print("Output:", result.stdout[-200:])
        else:
            print(f"✗ FAILED: Exit code {result.returncode}")
            print("STDERR:", result.stderr[-300:])
            print("STDOUT:", result.stdout[-300:])
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT: Direct call hung")
    except Exception as e:
        print(f"✗ ERROR: {e}")

def test_shell_call(runner_script, config_path):
    """Test subprocess call with shell=True"""
    cmd = f"{sys.executable} {runner_script} {config_path}"
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            cwd=str(Path.cwd())
        )
        
        if result.returncode == 0:
            print("✓ SUCCESS: Shell call worked")
            print("Output:", result.stdout[-200:])
        else:
            print(f"✗ FAILED: Exit code {result.returncode}")
            print("STDERR:", result.stderr[-300:])
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT: Shell call hung")
    except Exception as e:
        print(f"✗ ERROR: {e}")

def test_staggered_startup(runner_script, config_path):
    """Test multiple subprocesses with staggered startup"""
    print("Starting 3 processes with 5-second delays...")
    
    processes = []
    
    for i in range(3):
        cmd = [sys.executable, str(runner_script), config_path]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env['OMP_NUM_THREADS'] = '1'
        
        print(f"Starting process {i+1}...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path.cwd())
            )
            
            processes.append((process, f"process_{i+1}"))
            print(f"  Process {i+1} started with PID: {process.pid}")
            
            # Stagger the starts
            time.sleep(5)
            
        except Exception as e:
            print(f"  ERROR starting process {i+1}: {e}")
    
    # Monitor processes
    print("\nMonitoring processes...")
    start_time = time.time()
    
    while processes and (time.time() - start_time) < 120:  # 2 minute timeout
        completed = []
        
        for process, name in processes:
            poll_result = process.poll()
            if poll_result is not None:
                # Process completed
                try:
                    stdout, _ = process.communicate(timeout=1)
                    if poll_result == 0:
                        print(f"✓ {name} completed successfully")
                    else:
                        print(f"✗ {name} failed with exit code {poll_result}")
                        print(f"  Output: {stdout[-200:]}")
                except:
                    print(f"? {name} completed but couldn't read output")
                
                completed.append((process, name))
        
        # Remove completed processes
        for item in completed:
            processes.remove(item)
        
        if processes:
            time.sleep(2)
    
    # Kill any remaining processes
    for process, name in processes:
        print(f"⏰ Killing hanging process {name}")
        try:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        except:
            pass

if __name__ == "__main__":
    test_subprocess_variations()