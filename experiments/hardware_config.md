# Hardware and Software Configuration Template

Fill this in before running any experiments. Every latency and throughput
number in the paper must reference this exact configuration.

## Machine

| Field | Value |
|-------|-------|
| **Hostname** | |
| **OS** | e.g. Windows 11 Pro 23H2 / Ubuntu 22.04 LTS |
| **CPU** | e.g. AMD Ryzen 5 5600X (6C/12T, 3.7 GHz base) |
| **RAM** | e.g. 32 GB DDR4-3200 |
| **GPU** | e.g. NVIDIA RTX 3060 (12 GB) — note if unused during TFLite eval |
| **Storage** | e.g. NVMe SSD (dataset on fast local storage) |

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | | |
| TensorFlow | | |
| TFLite Runtime / LiteRT | | |
| OpenCV | | `opencv-python-headless` or `opencv-python` |
| NumPy | | |
| FastAPI | | |
| Uvicorn | | |
| scikit-learn | | |

## TFLite Interpreter Settings

| Parameter | Value |
|-----------|-------|
| `num_threads` | |
| `num_warmup_iterations` | 5 |
| `batch_size` | 32 |
| `input_resolution` | 96 × 96 |
| Quantization | Dynamic-range (int8 weights, float32 activations) |

## Git Commit

```
Commit: 643f6397dd43226695713a734f5658a858c992e9
Date:   (record the date you froze the commit)
```

## Notes

- Record ambient temperature if relevant (thermal throttling affects latency).
- Note whether power plan was set to "High Performance" (Windows) or
  `performance` governor (Linux).
- If running on a VM/cloud instance, record vCPU type and count.
