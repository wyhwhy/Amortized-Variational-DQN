"""
Implement vanilla DQN in Chainer
code adapted from Chainer tutorial
"""

from __future__ import print_function
from __future__ import division
import argparse
import collections
import copy
import random
import scipy
from scipy.stats import cauchy
import rank_based
import gym
import numpy as np
import os
import time
#import cupy as cp

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers,serializers

import HardMDP
n_actions=0

class QFunction(chainer.Chain):
    """Q-function represented by a MLP."""

    def __init__(self, obs_size, n_actions, n_units=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_actions*2)

    def __call__(self, x):
        """Compute Q-values of actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        qq = np.zeros((2*n_actions,n_actions))
        for i in range(n_actions):
            qq[2*i][i]=1
            #qq[2*i+1][i]=np.random.randn()
            qq[2*i+1][i]=scipy.stats.cauchy.rvs(loc=0, scale=1, size=1)
        #qq=cp.asarray(qq)
        return F.matmul(h,qq)
        #return chainer.Variable(np.matmul(h.data,qq))
        
class QFunctionplus(chainer.Chain):
    """Q-function represented by a MLP."""

    def __init__(self, obs_size, n_actions, n_units=100):
        super(QFunctionplus, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_units)
            self.l1 = L.Linear(n_units, n_units)
            self.l2 = L.Linear(n_units, n_actions*2)

    def __call__(self, x):
        """Compute Q-values of actions for given observations."""
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        '''
        h=h.data
        for i in range(n_actions):
            if h[0][2*i+1]>0.1:
                h[0][2*i+1]=0.1
        h=chainer.Variable(h)
        '''
        
        qq = np.zeros((2*n_actions,n_actions))
        for i in range(n_actions):
            qq[2*i][i]=1
            qq[2*i+1][i]=np.random.randn()
            #qq[2*i+1][i]=scipy.stats.cauchy.rvs(loc=0, scale=1, size=1)
        
        #qq=cp.asarray(qq)
        return F.matmul(h,qq)        

def get_greedy_action(Q, obs):
    """Get a greedy action wrt a given Q-function."""
    obs = Q.xp.asarray(obs[None], dtype=np.float32)
    with chainer.no_backprop_mode():
        q = (Q(obs).data[0])
    return int(q.argmax())


def mean_clipped_loss(y, t):
    return F.mean(F.huber_loss(y, t, delta=1.0, reduce='no'))


def update(D, Q, target_Q, opt, samples, gamma, target_type,indices):
    """Update a Q-function with given samples and a target Q-function."""
    xp = Q.xp
    obs = xp.asarray([sample[0] for sample in samples], dtype=np.float32)
    action = xp.asarray([sample[1] for sample in samples], dtype=np.int32)
    reward = xp.asarray([sample[2] for sample in samples], dtype=np.float32)
    done = xp.asarray([sample[3] for sample in samples], dtype=np.float32)
    obs_next = xp.asarray([sample[4] for sample in samples], dtype=np.float32)
    # Predicted values: Q(s,a)
    y = F.select_item((Q(obs)), action)
    # Target values: r + gamma * max_b Q(s',b)
    with chainer.no_backprop_mode():
        if target_type == 'dqn':
            next_q = F.max((target_Q(obs_next)), axis=1)
        elif target_type == 'double_dqn':
            next_q = F.select_item((target_Q(obs_next)),
                                   F.argmax((Q(obs_next)), axis=1))
        else:
            raise ValueError('Unsupported target_type: {}'.format(target_type))
        target = reward + gamma * (1 - done) * next_q
    D.update_priority(indices,(y-target).data)
    loss = mean_clipped_loss(y, target)
    Q.cleargrads()
    loss.backward()
    opt.update()
    #if D.record_size >295000:
     #   print(opt.eps)


def main():
    start =time.clock()
    parser = argparse.ArgumentParser(description='Chainer example: DQN')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Name of the OpenAI Gym environment')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Number of transitions in each mini-batch')
    parser.add_argument('--episodes', '-e', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='dqn_result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--target-type', type=str, default='dqn',
                        help='Target type', choices=['dqn', 'double_dqn'])
    parser.add_argument('--reward-scale', type=float, default=1e-2,
                        help='Reward scale factor')
    parser.add_argument('--replay-start-size', type=int, default=500,
                        help=('Number of iterations after which replay is '
                              'started'))
    parser.add_argument('--iterations-to-decay-epsilon', type=int,
                        default=5000,
                        help='Number of steps used to linearly decay epsilon')
    parser.add_argument('--min-epsilon', type=float, default=0.01,
                        help='Minimum value of epsilon')
    parser.add_argument('--target-update-freq', type=int, default=100,
                        help='Frequency of target network update')
    parser.add_argument('--record', action='store_true', default=True,
                        help='Record performance')
    parser.add_argument('--no-record', action='store_false', dest='record')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--random_initial_steps', type=int, default=30)
    args = parser.parse_args()

    # Initialize with seed
    seed = args.seed
    os.environ['CHAINER_SEED'] = str(seed)
    np.random.seed(seed)
    logdir = 'mVDQN-cauchypretrained-plus-gauss-rankbased/' + args.env + '/lr_' + str(args.lr) + 'episodes'+ str(args.episodes)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Initialize an environment
    env = gym.make(args.env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    obs_size = env.observation_space.low.size
    global n_actions
    n_actions = env.action_space.n

    reward_threshold = env.spec.reward_threshold
    if reward_threshold is not None:
        print('{} defines "solving" as getting average reward of {} over 100 '
              'consecutive trials.'.format(args.env, reward_threshold))
    else:
        print('{} is an unsolved environment, which means it does not have a '
              'specified reward threshold at which it\'s considered '
              'solved.'.format(args.env))

    # Initialize variables
    conf={'size': 10**6,
            'partition_num': 256,
            'batch_size': args.batch_size,
            'steps':env.spec.max_episode_steps*(1+args.episodes),
            'learn_start':args.replay_start_size}
    D = rank_based.Experience(conf)  # Replay buffer
    Rs = collections.deque(maxlen=10)  # History of returns
    iteration = 0
    #maxep = (args.episodes-200)
    maxep=2000
    #maxep=1500
    # Initialize a model and its optimizer
    Q = QFunction(obs_size, n_actions, n_units=args.unit)
    Qshadow = QFunctionplus(obs_size, n_actions, n_units=args.unit) #backup
    rmax = 0 #maximal reward
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        Q.to_gpu(args.gpu)
    target_Q = copy.deepcopy(Q)
    #opt = optimizers.SGD(eps=args.lr)
    opt = optimizers.Adam(eps=args.lr)
    opt.setup(Q)

    rrecord = []
    ttime=[]
    ar=[]
    vvisit=[]
    for episode in range(1+args.episodes):
        visit=np.zeros(128)
        obs = env.reset()
        done = False
        R = 0.0  # Return (sum of rewards obtained in an episode)
        timestep = 0

        while not done and timestep < env.spec.max_episode_steps:
            '''
            epsilon = 1 if len(D) < args.replay_start_size else \
                max(args.min_epsilon,
                    np.interp(
                        iteration,
                        [0, args.iterations_to_decay_epsilon],
                        [1, args.min_epsilon]))
            
            if np.random.rand() < 0.1:
                action = env.action_space.sample()
            else:
                action = get_greedy_action(Q, obs)
                '''
            action = get_greedy_action(Q, obs)

            # Execute an action
            new_obs, reward, done, _ = env.step(action)
            R += reward

            # Store a transition
            D.store((obs, action, reward * args.reward_scale, done, new_obs))
            obs = new_obs
            visit[int(sum(obs))-1]=visit[int(sum(obs))-1]+1
            # Sample a random minibatch of transitions and replay
            
            
            if D.record_size >= args.replay_start_size:
                
                #sample_indices = random.sample(range(D.record_size), args.batch_size)
                samples, sample_indices= D.sample(args.batch_size)
                update(D, Q, target_Q, opt, samples,
                       args.gamma, args.target_type, sample_indices)

            # Update the target network
            if iteration % args.target_update_freq == 0:
                target_Q = copy.deepcopy(Q)

            iteration += 1
            timestep += 1
            if iteration % 1000 ==0:
                D.rebalance()
            #D.rebalance()
        
        Rs.append(R)
        average_R = np.mean(Rs)
        rrecord.append(R)
        ar.append(average_R)
        vvisit.append(visit)
        ttime.append(round(time.clock()-start,2))
            
        if episode<=maxep:
        #if True:
            if episode==0:
                rmax=R
            if rmax<average_R:
                rmax=average_R
            if rmax==average_R:
                Qshadow=copy.deepcopy(Q)
        #if episode%maxep==0 and episode>0:
        if episode==maxep:
            Q=QFunctionplus(obs_size, n_actions, n_units=args.unit)
            #Q1=copy.deepcopy(Qshadow)
            serializers.save_npz('mvdqn.model', Qshadow)
            serializers.load_npz('mvdqn.model', Q)
            opt = optimizers.Adam(eps=args.lr)
            opt.setup(Q)
            #Q=QFunctionplus(obs_size, n_actions, n_units=args.unit)
            #Q=copy.deepcopy(Qshadow)
            #opt.eps=1e-12
            #opt.beta1=0.9
            #opt.beta2=0.999
            '''
            args.lr=args.lr*0.01
            opt = optimizers.Adam(eps=args.lr)
            opt.setup(Q)
            '''
            
        if episode>=maxep:
            opt.eps=args.lr/(1+0.9*(episode+1-maxep))
            
            #opt.setup(Q)
        print('episode: {} iteration: {} R: {} average_R: {}'.format(
              episode, iteration, R, average_R))

        #if episode % 100 ==0:
        
    #if episode % 10 ==0:
        #rrecord.append(R)
    np.save(logdir + '/AVDQN-rrecord_' + str(seed), rrecord)
    np.save(logdir + '/AVDQN-meanrrecord_' + str(seed), ar) 
    np.save(logdir + '/AVDQN-time_' + str(seed), ttime)
    np.save(logdir + '/AVDQN-visit_' + str(seed), vvisit)
    end = time.clock()
    print('Running time: %s Seconds'%(end-start))
    
if __name__ == '__main__':
    main()

'''
import time
start =time.clock()
end = time.clock()
print('Running time: %s Seconds'%(end-start))
'''