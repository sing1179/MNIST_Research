MNIST_Research
Algoverse Project: FFN_GeGLU vs FFN_ReLU on MNIST

This project investigates the performance of two Feed-Forward Network (FFN) variants — FFN_GeGLU and FFN_ReLU — on the MNIST dataset using PyTorch Lightning. The experiment evaluates whether the Gated Linear Unit (GLU) mechanism with GELU activation provides superior performance compared to the standard ReLU activation.

1. Claim
Hypothesis:
FFN_GeGLU performs better than FFN_ReLU when trained on MNIST.

This hypothesis builds on prior research (Shazeer, 2020) showing that gating mechanisms can improve representational power by allowing the network to selectively control information flow. The GeGLU architecture combines a GELU activation with a gating pathway, potentially making it more effective than simple ReLU activation, especially in small models.

2. Setup
Dataset:

MNIST (with 5,000 images reserved for validation)

Models Compared:

FFN_GeGLU:

FFN_GeGLU
(
𝑥
)
=
𝑊
𝑜
𝑢
𝑡
⋅
[
(
𝑊
𝑖
𝑛
⋅
𝑥
)
⊙
GELU
(
𝑊
𝑔
𝑎
𝑡
𝑒
⋅
𝑥
)
]
FFN_GeGLU(x)=W 
out
​
 ⋅[(W 
in
​
 ⋅x)⊙GELU(W 
gate
​
 ⋅x)]
FFN_ReLU:

FFN_ReLU
(
𝑥
)
=
𝑊
𝑜
𝑢
𝑡
⋅
ReLU
(
𝑊
𝑖
𝑛
⋅
𝑥
)
FFN_ReLU(x)=W 
out
​
 ⋅ReLU(W 
in
​
 ⋅x)
Training Configuration:

Epochs: 1 (One Epoch is All You Need)

Hidden Dimensions: [2, 4, 8, 16]

Batch Sizes: [8, 64]

Learning Rates: [1e-1, 1e-2, 1e-3, 1e-4]

Trials per configuration: 
𝑘
∈
{
2
,
4
,
8
}
k∈{2,4,8}

Selection Criterion: Best validation accuracy per hidden dimension/model

Bootstrap Resampling: 200 samples for error bar estimation

Frameworks & Tools:

PyTorch, PyTorch Lightning

NumPy, Matplotlib

3. Observations
k = 2 Trials
<img width="538" height="455" alt="Hidden_2_" src="https://github.com/user-attachments/assets/877a7cdb-20dd-47ff-821b-ba86808bcec5" />
Observation: At smaller hidden dimensions, FFN_GeGLU outperforms FFN_ReLU.

k = 4 Trials
<img width="537" height="456" alt="Hidden_4" src="https://github.com/user-attachments/assets/a1c6a15f-fe99-40da-ab08-7c50fda802a6" />
Observation: The performance gap narrows, but FFN_GeGLU remains slightly more stable.

k = 8 Trials
<img width="538" height="476" alt="Hidden_8" src="https://github.com/user-attachments/assets/d09e628a-f594-4c36-a96e-6acf8912f731" />
Observation: At larger hidden dimensions, both models perform similarly, but FFN_ReLU shows occasional dips likely due to poor hyperparameter choices.

4. Conclusion
The results support the claim that FFN_GeGLU is generally better than FFN_ReLU, especially in low-parameter regimes (hidden dimensions 2 and 4). Its gating mechanism offers stable performance across different random search trials, while FFN_ReLU is more sensitive to hyperparameter variations.

5. Limitations
Training Duration: Only one epoch per trial — results may change with longer training.

Hyperparameter Search: Limited to random search; optimal configurations may not always be found.

Dataset Scope: Only tested on MNIST; results may not generalize to more complex datasets.
