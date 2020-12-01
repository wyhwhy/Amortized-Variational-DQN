# Amortized-Variational-DQN
Code for paper 'Amortized Variational Deep Q Network' on http://arxiv.org/abs/2011.01706, accepted to Deep RL workshop NeurIPS 2020.

## Example for running the code:
  * 1.python m-VDQN-cauchypretrained-plus-gauss-rankbased.py --env CartPole-v1 --episodes 1500 --reward-scale 1e-1 --lr 1e-5 --batch-size 128 --replay-start-size 4096 --seed 20 --gamma 0.99
  * 2.python main_noisynet.py --env 128NDeterministicChain-v0 --episodes 450 --gamma 1 --seed 20

## Acknowledgement
Thanks Yunhao Tang for his work on Variational Deep Q Network and MDP chain environment HardMDP on https://github.com/robintyh1/Variational-DQN. Thanks for Damcy's work on Prioritized Experience Replay (which was modified) on https://github.com/Damcy/prioritized-experience-replay.

## Citation
If you use our code or refer to our paper, please cite the following. Thanks!
Haotian Zhang, Yuhao Wang, Jianyong Sun, and Zongben Xu. "Amortized Variational Deep Q Network." arXiv preprint arXiv:2011.01706 (2020). Deep Reinforcement Learning Workshop, NIPS, 2020.
