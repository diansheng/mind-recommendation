# mind-recommendation
Explore various approaches to solve microsoft news recommendation problem.


##  NRSM

In microsoft, the best single model is NRMS ([Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671.pdf) ). In NRMS, training is formulated as a multi-class classification problem with a fixed number of negative samples.


## Recommendation System

A framework that allows to train, evaluate and predict various recommendation algorithms. 

#### Components
- Dataset
  - User behavior sequence(in order) or set(not in order)
  - User side information, including user geographical information and statistical feature
  - Item/Entity side information, including item content information and statistical feature
- Algorithm
  - Rule-based
  - Machine learning based
- System
  - Train and inference. May make use of multiple algorithms.

## Usage


## Example

MIND news recommendation public dataset. 

- Dataset:
  - User behavior sequence
  - User side information
