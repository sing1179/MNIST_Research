MNIST_Research
Algoverse Project: FFN_GeGLU vs FFN_ReLU on MNIST

This project investigates the performance of two Feed-Forward Network (FFN) variants — FFN_GeGLU and FFN_ReLU — on the MNIST dataset using PyTorch Lightning. The experiment evaluates whether the Gated Linear Unit (GLU) mechanism with GELU activation provides superior performance compared to the standard ReLU activation.

1. Claim
Hypothesis:
FFN_GeGLU performs better than FFN_ReLU when trained on MNIST.

This hypothesis builds on prior research (Shazeer, 2020) showing that gating mechanisms can improve representational power by allowing the network to selectively control information flow. The GeGLU architecture combines a GELU activation with a gating pathway, potentially making it more effective than simple ReLU activation, especially in small models.

2. Setup
Dataset:

MNIST 

Models Compared:

FFN_GeGLU and FFN_ReLU 


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

Bootstrap Resampling: 50 samples for error bar estimation

Frameworks & Tools:

PyTorch, PyTorch Lightning

NumPy, Matplotlib

3. Observations
k = 2 Trials

<img width="538" height="455" alt="Hidden_2_" src="https://github.com/user-attachments/assets/877a7cdb-20dd-47ff-821b-ba86808bcec5" />

Observation: ReLU significantly outperforms GeGLU at hidden dimensions 8 and slightly at 2, while GeGLU edges ahead at dimension 16.

GeGLU performance drops sharply at hidden dimension 8, indicating instability with limited trials.

k = 4 Trials

<img width="537" height="456" alt="Hidden_4" src="https://github.com/user-attachments/assets/a1c6a15f-fe99-40da-ab08-7c50fda802a6" />

Observation: Both models perform very similarly across dimensions, with GeGLU slightly ahead at dimensions 2 and 8.

Stability improves for both models compared to k=2.

k = 8 Trials

<img width="538" height="476" alt="Hidden_8" src="https://github.com/user-attachments/assets/d09e628a-f594-4c36-a96e-6acf8912f731" />

Observation: GeGLU consistently outperforms ReLU at all hidden dimensions, with the largest gap at lower dimensions (2 and 4).

Higher trial count smooths out instability seen in k=2.



4. Conclusion
At low hidden dimensions (2, 4), GeGLU shows a clearer advantage as k increases, suggesting the gating mechanism provides more efficient representation in low-capacity models.

At high hidden dimensions (8, 16), performance is comparable, but GeGLU still edges ahead in higher-trial experiments.

The claim that FFN_GeGLU is better than FFN_ReLU is not strongly supported for k=2, but becomes more valid for k=4 and strongly valid for k=8.

5. Limitations
Training Duration: Only one epoch per trial — results may differ with longer training.

Hyperparameter Search: Limited to random search; grid search or Bayesian optimization might yield different results.

Dataset Scope: Only tested on MNIST; results may not generalize to more complex datasets.
