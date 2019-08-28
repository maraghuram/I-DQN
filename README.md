## Towards Better Interpretability in Deep Q-Networks
[[Paper]](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4377)[[Arxiv]](https://arxiv.org/abs/1809.05630)
 
Original codebase for training agents and analysing trained models.
 


##### Abstract

Deep reinforcement learning techniques have demonstrated superior performance in a wide variety of environments. As improvements in training algorithms continue at a brisk pace, theoretical or empirical studies on understanding what these networks seem to learn, are far behind. In this paper we propose an interpretable neural network architecture for Q-learning which provides a global explanation of the model's behavior using key-value memories, attention and reconstructible embeddings. With a directed exploration strategy, our model can reach training rewards comparable to the state-of-the-art deep Q-learning models. However, results suggest that the features extracted by the neural network are extremely shallow and subsequent testing using out-of-sample examples shows that the agent can easily overfit to trajectories seen during training. 

#### Requirements
___
- Python 3.6.6
- Pytorch, torch==0.4.0
- TensorboardX, tensorboardX==1.4

#### Citation
___
```
@inproceedings{annasamy2019towards,
  title={Towards better interpretability in deep q-networks},
  author={Annasamy, Raghuram Mandyam and Sycara, Katia},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={4561--4569},
  year={2019}
}
```