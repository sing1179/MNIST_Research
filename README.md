# MNIST_Research
Algoverse Project: FFN_GeGLU vs FFN_ReLU on MNIST
This project investigates the performance of two Feed-Forward Network (FFN) variants, FFN_GeGLU and FFN_ReLU, on the MNIST dataset using PyTorch Lightning. The experiment is designed to evaluate whether the Gated Linear Unit (GLU) mechanism with GELU activation provides superior performance compared to the standard ReLU activation.

1. Claim
The claim tested is:

FFN_GeGLU performs better than FFN_ReLU when trained on MNIST.

This hypothesis is grounded in prior research on GLUs (Shazeer, 2020), which suggests that gating mechanisms improve representational power by allowing the network to selectively control the flow of information.

Training Configuration
Epochs: 1 (One Epoch is All You Need)

Hidden Dimensions: [2, 4, 8, 16]

Batch Sizes: [8, 64]

Learning Rates: [1e-1, 1e-2, 1e-3, 1e-4]

Random Search Trials: 
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


Hyperparameter Search
For each combination of model type and hidden dimension:

Run k random hyperparameter configurations.

Select the configuration with maximum validation accuracy.

Report test accuracy using bootstrap resampling (200 samples) to compute error bars.

2. Results
Three sets of plots were generated, one for each 
𝑘
k value (2, 4, 8), showing Test Accuracy vs Hidden Dimension for both models.


<img width="538" height="455" alt="Hidden_2_" src="https://github.com/user-attachments/assets/877a7cdb-20dd-47ff-821b-ba86808bcec5" />

k = 2 Trials

<img width="537" height="456" alt="Hidden_4" src="https://github.com/user-attachments/assets/a1c6a15f-fe99-40da-ab08-7c50fda802a6" />

k = 4 Trials

<img width="538" height="476" alt="Hidden_8" src="https://github.com/user-attachments/assets/d09e628a-f594-4c36-a96e-6acf8912f731" />

k = 8 Trials

3. Analysis
Low Hidden Dimensions (2, 4):
FFN_GeGLU generally outperforms FFN_ReLU, confirming that the gating mechanism is more effective when model capacity is small.

Higher Hidden Dimensions (8, 16):
Performance differences narrow, with both models achieving similar accuracies. However, FFN_ReLU displayed greater sensitivity to poor hyperparameter choices, resulting in occasional dips in performance.

Stability Across Trials:
FFN_GeGLU’s accuracy curve was smoother and more predictable across different 
𝑘
k values, while FFN_ReLU showed occasional erratic behavior.

4. Conclusion
The experimental results support the claim that FFN_GeGLU is better than FFN_ReLU, particularly in low-parameter settings. The gating mechanism in FFN_GeGLU helps it maintain stable, competitive performance across hidden dimensions and random search trials, making it a more reliable architecture under constrained conditions.

5. Key Concepts & References
Key Concepts:

Feed-Forward Networks (FFNs)

Gated Linear Units (GLUs) and GeGLU activation

PyTorch Lightning training loops

Random hyperparameter search

Bootstrap confidence intervals

