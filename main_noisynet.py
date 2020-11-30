"""
Implement Variational DQN in Edward and Tensorflow
"""

import tensorflow as tf
import edward as ed
from edward.models import Normal
import numpy as np
import gym
import copy
import os
import argparse
import time
from models_noisynet import VariationalQNetwork,NoiseSampler,update_target
from utils import ReplayBuffer
from chainer import serializers

import HardMDP


def main():
	start =time.clock()   
	parser = argparse.ArgumentParser(description='VDQN')
	parser.add_argument('--seed',type=int,default=100)
	parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Name of the OpenAI Gym environment')
	parser.add_argument('--logdir',type=str,default='')
	parser.add_argument('--episodes',type=int,default=200)
	parser.add_argument('--target-update-period',type=int,default=100)
	parser.add_argument('--lr',type=float,default=1e-3)
	parser.add_argument('--gamma',type=float,default=.99)
	args = parser.parse_args()

	#### HYPERPARAMETERS
	episodes = args.episodes #1000

	envname = args.env

	seed = args.seed
	tf.set_random_seed(seed)
	np.random.seed(seed)

	hiddendict = [100,100]
	sigma = 0.01
	Wpriorsigma = [10000] * 2
	bpriorsigma = [10000] * 2
	batchsize = 64
	buffersize = 1000000
	initialsize = 500
	tau = 1.0
	target_update_period = args.target_update_period
	lr_VI = args.lr
	gamma = args.gamma
	totalstep = 0
	reward_scale = 1
	############
	#### MAIN ITERATIONS
	###########
	logdir = args.logdir + 'noisynet/' + envname  + '/lr_' + str(args.lr) + '_episodes' + str(args.episodes) 
	
	if not os.path.exists(logdir):
		os.makedirs(logdir)
		
	
	with tf.Session() as sess:
		
		#saver=tf.train.Saver()
		### INITIALIZATION
		env = gym.make(envname)
		obssize = env.observation_space.low.size
		actsize = env.action_space.n
		replaybuffer = ReplayBuffer(buffersize)
		Qactionnet = VariationalQNetwork(obssize,actsize,hiddendict,sess=sess,scope='principle',optimizer=tf.train.AdamOptimizer(lr_VI))
		Qtargetnet = VariationalQNetwork(obssize,actsize,hiddendict,sess=sess,scope='target')
		noisesampler = NoiseSampler(Qactionnet.Wshape,Qactionnet.bshape)

		sess.run(tf.global_variables_initializer())

		update_target(Qtargetnet,Qactionnet)

		### RECORD
		#VIlossrecord = []
		Bellmanlossrecord = []
		rewardrecord = []
		ar=[]
		vvisit=[]
        
		#print([obssize] + hiddendict + [actsize])
		### ITERATIONS
		for episode in range(episodes+1):
			visit=np.zeros(128)
			# start
			obs = env.reset()
			done = False

			rsum = 0
			while not done:
				# sample a noise and compute 
				Wnoise,bnoise = noisesampler.sample(1)
				# compuet Q value
				Qvalue = Qactionnet.compute_Qvalue(obs[None],Wnoise,bnoise)
				# select action
				action = np.argmax(Qvalue.flatten())
				# step
				nextobs,reward,done,_ = env.step(action)
				# record experience
				done_ = 1 if done else 0
				reward_ = reward * reward_scale
				experience = [(obs,action,reward_,done_,nextobs)]
				# append experience to buffer
				replaybuffer.append(experience)
				replaybuffer.popleft()
				# update
				obs = nextobs
				totalstep += 1
				rsum += reward
				visit[int(sum(obs))-1]=visit[int(sum(obs))-1]+1
				if replaybuffer.currentsize >= initialsize:
					# sample minibatch 
					batch_obs,batch_act,batch_reward,batch_done,batch_nextobs = replaybuffer.sample(batchsize)
					# sample noise for computing target
					Wnoise,bnoise = noisesampler.sample(batchsize)
					# compute target value
					Qall = Qtargetnet.compute_Qvalue(batch_nextobs,Wnoise,bnoise)
					Qtarget = gamma * np.max(Qall,axis=1) * (1-batch_done) + batch_reward
					# udpate principle network by VI
                    
					#VIloss = Qactionnet.train_on_sample(batch_obs,batch_act,Qtarget,episode)
					
                    # comptue bellman error loss
					#Wnoise_new,bnoise_new = noisesampler.sample(batchsize)
					#Wnoise_new,bnoise_new = Wnoise,bnoise
					#Qpred = Qactionnet.compute_Qvalue(batch_obs,Wnoise_new,bnoise_new)
					#Qpredact = Qpred[np.arange(batchsize),batch_act]
					#Bellmanloss = np.mean((Qpredact - Qtarget)**2)
					Qactionnet.train_on_sample(batch_obs,batch_act,Qtarget,Wnoise,bnoise)
                    #a,b,c,d = Qactionnet.get_variables()
					#print('Wmu',a,'Wrho',b,'bmu',c,'brho',d)
					#print(Qpredact,Qtarget)
					# record	
					#print('bellmanerror',Bellmanloss)
					#raise ValueError		
					#VIlossrecord.append(VIloss['loss'])
					#Bellmanlossrecord.append(Bellmanloss)

				if (totalstep+1) % target_update_period == 0:
					update_target(Qtargetnet,Qactionnet)
					#Qtargetnet = copy.deepcopy(Qactionnet)
					#print("update target")

				if done:
					break
				
			# record
			rewardrecord.append(rsum)
			ar.append(np.mean(rewardrecord[-10:]))
			vvisit.append(visit)
			### TRAIN

			#meanVIloss = np.mean(VIlossrecord[-10:]) if len(VIlossrecord)>10 else np.float('nan')
			#meanbellmanloss = np.mean(Bellmanlossrecord[-10:]) if len(Bellmanlossrecord)>10 else np.float('nan')
			#meanreward = np.mean(rewardrecord[-10:])

			#print("episode %d buffer size %d meanVIloss %f meanbellmanloss %f meanreward %f" %(episode,replaybuffer.currentsize,meanVIloss,meanbellmanloss,meanreward))
			
			#saver.save(sess,'d:\\model\\net_model')
            
			if episode%10==0:
				#np.save(logdir+'/VDQN-VIloss_'+str(seed),VIlossrecord)
				np.save(logdir+'/noisynet-bellmanloss_'+str(seed),Bellmanlossrecord)
				np.save(logdir+'/noisynet-reward_'+str(seed),rewardrecord)
				np.save(logdir+'/noisynet-meanreward_'+str(seed),ar)
				np.save(logdir+'/noisynet-visit_'+str(seed),vvisit)

		end = time.clock()
		print('Running time: %s Seconds'%(end-start))

if __name__ == '__main__':
	main()










	

