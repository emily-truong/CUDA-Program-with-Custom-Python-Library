# CUDA-Program-with-Custom-Python-Library

Two programs: **CPU** (`matrix_cpu.c`) and **GPU** (`matrix_gpu.cu`).

## CPU (Mac or any machine)

```bash
gcc matrix_cpu.c -o matrix_cpu -O2
./matrix_cpu 512
./matrix_cpu 1024
./matrix_cpu 2048
```

## GPU (NVIDIA GPU + CUDA toolkit)

```bash
nvcc matrix_gpu.cu -o matrix_gpu -O2
./matrix_gpu 512
./matrix_gpu 1024
./matrix_gpu 2048
```

Check CUDA:

```bash
nvcc --version
nvidia-smi
```
