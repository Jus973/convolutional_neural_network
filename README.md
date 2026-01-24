# ATC-ASR Spectrogram Classifier

A from-scratch implementation of a Convolutional Neural Network (CNN) for classifying Air Traffic Control (ATC) audio. Bypasses high-level frameworks to implement the underlying linear algebra and calculus. Optimize with GPU kernels.

## Architecture & Design

Input:
(N, 1, 64, 400)

Block 1
Conv2D: 1 → 16, 3×3, stride 1, pad 1 →
(N, 16, 64, 400)
ReLU
MaxPool2D: 2×2 →
(N, 16, 32, 200)

Block 2
Conv2D: 16 → 32, 3×3, pad 1 →
(N, 32, 32, 200)
ReLU
MaxPool2D: 2×2 →
(N, 32, 16, 100)

Block 3
Conv2D: 32 → 64, 3×3, pad 1 →
(N, 64, 16, 100)
ReLU
MaxPool2D: 2×2 →
(N, 64, 8, 50)

Classifier
Flatten →
(N, 64 × 8 × 50 = 25600)
Dense: 25600 → 128, ReLU
Dense: 128 → 10
Softmax

## Optimization Roadmap

### Phase 1: Mathematical Optimization
- **im2col**: Lowering convolutions to dense matrix multiplications (GEMM) to improve cache locality.
- **Quantization**: Reducing precision to INT8 for increased arithmetic throughput.

### Phase 2: CUDA Implementation (A100 Target)
- **Memory Management**: Using constant memory for kernels and shared memory for tiling.
- **Precision**: Implementing `__half2` (FP16) to leverage Tensor Core acceleration.
- **Concurrency**: Overlapping H2D/D2H transfers with compute using CUDA streams.
- **Profiling**: Identifying compute vs. memory bottlenecks using `nsys` and `ncu`.


https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset
