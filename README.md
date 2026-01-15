

Intended Structure:
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




https://huggingface.co/datasets/jacktol/ATC-ASR-Dataset
