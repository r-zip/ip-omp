# Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI
**Aditya Chattopadhyay¹, Ryan Pilgrim¹, and René Vidal²** <br>

**¹Johns Hopkins University, USA, `{achatto1,rpilgri1}@jhu.edu`**

**²University of Pennsylvania, USA, `vidalr@upenn.edu`**

This repository contains code to accompany [*Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI (NeurIPS 2023)*](https://openreview.net/forum?id=CAF4CnUblx).

## Overview
<p align="center">
<img src="./assets/explainable_examples.png" alt="predictions" width="800"/>
</p>

Information Pursuit (IP) is a classical active testing algorithm for predicting an output by sequentially and greedily querying the input in order of information gain. However, IP is computationally intensive since it involves estimating mutual information in high-dimensional spaces. This paper explores Orthogonal Matching Pursuit (OMP) as an alternative to IP for greedily selecting the queries. OMP is a classical signal processing algorithm for sequentially encoding a signal in terms of dictionary atoms chosen in order of correlation gain. In each iteration, OMP selects the atom that is most correlated with the signal residual (the signal minus its reconstruction thus far). Our first contribution is to establish a fundamental connection between IP and OMP, where we prove that IP with random projections of dictionary atoms as queries "almost" reduces to OMP, with the difference being that IP selects atoms in order of normalized correlation gain. We call this version IP-OMP and present simulations indicating that this difference does not have any appreciable effect on the sparse code recovery rate of IP-OMP compared to that of OMP for random Gaussian dictionaries. Inspired by this connection, our second contribution is to explore the utility of IP-OMP for generating explainable predictions, an area in which IP has recently gained traction. More specifically, we propose a simple explainable AI algorithm which encodes an image as a sparse combination of semantically meaningful dictionary atoms that are defined as text embeddings of interpretable concepts. The final prediction is made using the weights of this sparse combination, which serve as an explanation. Empirically, our proposed algorithm is not only competitive with existing explainability methods but also computationally less expensive.

## Setup
Coming soon.

## Sparse Recovery Experiments
To execute the sparse recovery experiments, run
```
./script/sparse_recovery_experiments.sh
```
from the repository root directory. To plot the results, run
```
./script/sparse_recovery_plots.sh
```

## CLIP-IP-OMP Experiments
Coming soon.

## License
This repository is MIT-licensed. See [LICENSE](./LICENSE) for details.

## Cite
```
@inproceedings{
chattopadhyay2023information,
title={Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable {AI}},
author={Aditya Chattopadhyay and Ryan Pilgrim and Rene Vidal},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=CAF4CnUblx}
}
```
