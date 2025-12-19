# GPU Stress Test

Simple python script that will stress test whatever GPU you have installed. It will also show a log while running so you can monitor metrics.

## How to Run
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/IIEleven11/Simplepod_Stress_Test.git
2. **Navigate to the Directory**:
    ```bash
    cd Simplepod_Stress_Test
    ```
3. **Install Dependencies**:
    Make sure you have Python and pip installed. Then run:
    
    Install UV: ``` curl -LsSf https://astral.sh/uv/install.sh | sh ```
    
    Create venv: ``` uv venv --python 3.12 ```

    Activate venv: ``` source .venv/bin/activate ```
    
    Install torch: ``` uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu129 ```
    

3. **Run the Stress Test**:
    ```bash
    python gpu_stress_test.py --duration <integer>  --monitor-interval <integer>
    ```

## Arguments
- `--duration`: Duration of the test it defaults to 60 seconds.
- `--monitor-interval`: Interval the log should update metrics at. Default is two seconds..

### Example
Run a stress test for 120 seconds and log GPU stats every 10 seconds:
```bash
python gpu_stress_test.py --duration 120 --monitor-interval 10
```

