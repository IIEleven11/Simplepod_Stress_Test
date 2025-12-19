import torch
import time
import argparse
import threading
import sys
import subprocess
import shutil
import signal
import os

# Global flag to control monitoring thread
stop_monitoring = False

def signal_handler(signum, frame):
    """
    Handle signals like SIGTSTP (Ctrl+Z) to prevent suspension and ensure clean exit.
    """
    print("\n\nReceived signal (Ctrl+C or Ctrl+Z). Exiting cleanly...")
    global stop_monitoring
    stop_monitoring = True
    # Force exit to ensure threads are killed
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTSTP'):
    signal.signal(signal.SIGTSTP, signal_handler)

def get_gpu_metrics():
    """
    Queries nvidia-smi for GPU metrics.
    Returns a list of dictionaries containing metrics for each GPU.
    """
    try:
        # Check if you have the driver installed then we look at the metrics
        if not shutil.which("nvidia-smi"):
            return None

        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        
        metrics = []
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 7:
                metrics.append({
                    "index": parts[0],
                    "util_gpu": float(parts[1]),
                    "mem_used": float(parts[2]),
                    "mem_total": float(parts[3]),
                    "power_draw": float(parts[4]),
                    "power_limit": float(parts[5]),
                    "temp": float(parts[6])
                })
        return metrics
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return None

def monitor_gpus(interval=2):
    print("\nStarting")
    print(f"{'GPU':<4} | {'Util':<6} | {'Memory (MB)':<15} | {'Power (W)':<15} | {'Temp (C)':<8}")
    print("-" * 60)
    
    while not stop_monitoring:
        metrics = get_gpu_metrics()
        if metrics:
            timestamp = time.strftime("%H:%M:%S")
            print(f"\n--- Metrics at {timestamp} ---")
            print(f"{'GPU':<4} | {'Util':<6} | {'Memory (MB)':<15} | {'Power (W)':<15} | {'Temp (C)':<8}")
            
            for m in metrics:
                mem_str = f"{int(m['mem_used'])} / {int(m['mem_total'])}"
                power_str = f"{m['power_draw']:.1f} / {m['power_limit']:.1f}"
                print(f"{m['index']:<4} | {m['util_gpu']:>5.1f}% | {mem_str:<15} | {power_str:<15} | {m['temp']:>6.1f}")
        
        time.sleep(interval)

def stress_gpu(gpu_id, duration, target_vram_gb=None):
    """
    Runs a stress test on a specific GPU.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory
        
        print(f"GPU {gpu_id}: {props.name}")
        print(f"GPU {gpu_id}: Total Memory: {total_memory / 1024**3:.2f} GB")

        # Check if VRAM is already heavily used (e.g. by a suspended process)
        # We check nvidia-smi metrics because torch.cuda.mem_get_info might report free memory 
        # that is technically "free" but practically unusable if another process is holding it in a weird state.
        metrics = get_gpu_metrics()
        current_usage_mb = 0
        if metrics:
            for m in metrics:
                if str(m['index']) == str(gpu_id):
                    current_usage_mb = m['mem_used']
                    break
        
        print(f"GPU {gpu_id}: Current VRAM usage: {current_usage_mb} MB")
        
        # If more than 20% of memory is used, we assume it's dirty and skip filler to ensure compute runs
        total_mem_mb = total_memory / 1024**2
        skip_filler = False
        if current_usage_mb > (total_mem_mb * 0.2):
            print(f"GPU {gpu_id}: WARNING - High memory usage detected! Skipping filler allocation to ensure compute runs.")
            skip_filler = True

        # Get actual free memory reported by driver
        free_memory, total_memory_info = torch.cuda.mem_get_info(device)
        print(f"GPU {gpu_id}: Driver reports Free Memory: {free_memory / 1024**3:.2f} GB")

        # Calculate how much memory to allocate
        # We want to leave enough space for the compute loop and system overhead.
        # 16k x 16k float32 matrix is 1GB. We need 3 of them (A, B, C) = 3GB.
        # Plus some overhead. Let's leave 5GB free.
        compute_reserve = 5 * 1024**3 
        
        alloc_memory = free_memory - compute_reserve
        
        if target_vram_gb:
             # If target is specified, we try to hit that target, but capped by available free memory
             target_bytes = int(target_vram_gb * 1024**3)
             if target_bytes < alloc_memory:
                 alloc_memory = target_bytes

        if not skip_filler and alloc_memory > 0:
            print(f"GPU {gpu_id}: Attempting to allocate {alloc_memory / 1024**3:.2f} GB filler...")
            num_elements = alloc_memory // 4
            try:
                # Use empty first, then fill
                filler = torch.empty(num_elements, dtype=torch.float32, device=device)
                # Fill with a value to force physical allocation
                filler.fill_(1.0) 
                print(f"GPU {gpu_id}: Successfully allocated and filled filler tensor.")
            except RuntimeError as e:
                # print(f"GPU {gpu_id}: Failed to allocate filler: {e}")
                print(f"GPU {gpu_id}: Could not allocate extra filler (VRAM might already be full). Proceeding to compute...")
                # If filler fails, we proceed to compute anyway, just with less VRAM usage
        else:
            if skip_filler:
                print(f"GPU {gpu_id}: Skipped filler due to high existing usage.")
            else:
                print(f"GPU {gpu_id}: Not enough free memory for filler, proceeding to compute only.")

        # Compute tensors
        # 16384x16384 float32 is 1GB.
        N = 16384 
        print(f"GPU {gpu_id}: Starting compute loop with matrix size {N}x{N}")
        
        try:
            a = torch.randn(N, N, device=device, dtype=torch.float32)
            b = torch.randn(N, N, device=device, dtype=torch.float32)
        except RuntimeError as e:
             print(f"GPU {gpu_id}: Failed to allocate compute tensors: {e}")
             return
        
        start_time = time.time()
        
        print(f"GPU {gpu_id}: Running stress test...")
        while True:
            if duration > 0 and (time.time() - start_time) > duration:
                break
            
            # Perform matrix multiplication
            c = torch.mm(a, b)
            
            # Add a dependency to prevent optimization and keep memory bus active
            a.add_(c, alpha=0.0001)
            
            # Synchronize to ensure the GPU is actually finishing the work
            torch.cuda.synchronize()

    except Exception as e:
        print(f"GPU {gpu_id}: Error occurred: {e}")
    finally:
        print(f"GPU {gpu_id}: Finished.")

def main():
    global stop_monitoring
    parser = argparse.ArgumentParser(description="GPU Stress Test Script")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the test in seconds (0 for infinite)")
    parser.add_argument("--target-vram", type=float, default=None, help="Target VRAM to fill in GB (default: max available)")
    parser.add_argument("--monitor-interval", type=int, default=2, help="Interval in seconds to print GPU metrics")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("No CUDA devices found!")
        sys.exit(1)
        
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA devices.")
    
    monitor_thread = threading.Thread(target=monitor_gpus, args=(args.monitor_interval,))
    monitor_thread.start()
    
    threads = []
    try:
        for i in range(num_gpus):
            t = threading.Thread(target=stress_gpu, args=(i, args.duration, args.target_vram))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping...")
    finally:
        stop_monitoring = True
        monitor_thread.join()
        
    print("All stress tests completed.")

if __name__ == "__main__":
    main()
