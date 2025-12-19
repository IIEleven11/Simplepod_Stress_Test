import torch
import time
import argparse
import threading
import sys
import subprocess
import shutil

stop_monitoring = False

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

        # Calculate how much memory to allocate
        reserved = 1 * 1024**3 
        alloc_memory = total_memory - reserved
        
        if target_vram_gb:
             alloc_memory = min(alloc_memory, int(target_vram_gb * 1024**3))

        print(f"GPU {gpu_id}: Attempting to allocate {alloc_memory / 1024**3:.2f} GB")

        compute_buffer_size = 2 * 1024**3
        filler_size = alloc_memory - compute_buffer_size
        
        if filler_size > 0:
            num_elements = filler_size // 4
            try:
                filler = torch.empty(num_elements, dtype=torch.float32, device=device)
                filler.normal_()
                print(f"GPU {gpu_id}: Allocated {filler_size / 1024**3:.2f} GB filler tensor")
            except RuntimeError as e:
                print(f"GPU {gpu_id}: Failed to allocate filler: {e}")
                return

        N = 8192 
        print(f"GPU {gpu_id}: Starting compute loop with matrix size {N}x{N}")
        
        a = torch.randn(N, N, device=device, dtype=torch.float32)
        b = torch.randn(N, N, device=device, dtype=torch.float32)
        
        start_time = time.time()
        
        while True:
            if duration > 0 and (time.time() - start_time) > duration:
                break
            
            c = torch.mm(a, b)

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
