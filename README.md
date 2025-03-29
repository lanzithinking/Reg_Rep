# Bayesian Regularization of Latent Representation

## Chukwudi Paul Obite, Zhi Chang, Keyan Wu, Shiwei Lan

## The Thirteenth International Conference on Learning Representations (ICLR) 2025, EXPO Singapore

### Please cite the [paper](https://openreview.net/pdf?id=VOoJEQlLW5).


The effectiveness of statistical and machine learning methods depends on how well data features are characterized. Developing informative and interpretable latent representations with controlled complexity is essential for visualizing data structure and for facilitating efficient model building through dimensionality reduction. Latent variable models, such as Gaussian Process Latent Variable Models (GP-LVM), have become popular for learning complex, nonlinear representations as alternatives to Principal Component Analysis (PCA). In this paper, we propose a novel class of latent variable models based on the recently introduced Q-exponential process (QEP), which generalizes GP-LVM with a tunable complexity parameter, $q>0$. Our approach, the \emph{Q-exponential Process Latent Variable Model (QEP-LVM)}, subsumes GP-LVM as a special case when $q=2$, offering greater flexibility in managing representation complexity while enhancing interpretability. To ensure scalability, we incorporate sparse variational inference within a Bayesian training framework. We establish connections between QEP-LVM and probabilistic PCA, demonstrating its superior performance through experiments on datasets such as the Swiss roll, oil flow, and handwritten digits.


**Requirement***

- [PyTorch](https://pytorch.org)

- [GPyTorch](https://gpytorch.ai)