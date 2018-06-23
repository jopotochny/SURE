from builtins import range
import os
import argparse
import random
from collections import namedtuple
import pygame
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pygame import surfarray, key, event, display, time, locals
import Computer
import pandas as pd
import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import argparse
import numpy as np
import random
import os
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import gym
import time
from collections import deque

parser = argparse.ArgumentParser(description='QLearningPong')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to latest checkpoint')

VALID_ACTION = [0, 3, 4]
GAMMA = 0.99
epsilon = 0.5
update_step = 1000
memory_size = 20000
max_epoch = 1000000
batch_size = 64
save_path = './tmp'

device = torch.device("cpu")
# device = torch.device("cuda")

# Variables
with torch.no_grad():
	var_phi = autograd.Variable(torch.Tensor(1, 4, 84, 84)).to(device)
# For training
var_batch_phi = autograd.Variable(torch.Tensor(batch_size, 4, 84, 84)).to(device)
var_batch_a = autograd.Variable(torch.LongTensor(batch_size, 1), requires_grad=False).to(device)
var_batch_r = autograd.Variable(torch.Tensor(batch_size, 1)).to(device)
var_batch_phi_next = autograd.Variable(torch.Tensor(batch_size, 4, 84, 84)).to(device)
var_batch_r_mask = autograd.Variable(torch.Tensor(batch_size, 1), requires_grad=False).to(device)
def save_statistic(ylabel, nums, std=None, save_path=None):

	n = np.arange(len(nums))

	plt.figure()
	plt.plot(n, nums)
	if std is not None:
		nums = np.array(nums)
		std = np.array(std)
		plt.fill_between(n, nums+std, nums-std, facecolor='blue', alpha=0.1)
	plt.ylabel(ylabel)
	plt.xlabel('Episodes')
	plt.savefig(save_path + '/' + ylabel + '.png')
	plt.close()

def sample_action(env, agent, var_phi, epsilon):

	if random.uniform(0,1) > epsilon:
		phi = env.current_phi
		var_phi.data.copy_(torch.from_numpy(phi))

		q_values = agent(var_phi)
		max_q, act_index = q_values.max(dim=0)
		act_index = np.asscalar(act_index.data.cpu().numpy())
	else:
		act_index = random.randrange(3)

	return act_index
class DQN(nn.Module):

	def __init__(self):

		super(DQN, self).__init__()

		self.main = nn.Sequential(
			nn.Conv2d(4, 32, 8, 4, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, 4, 2, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 512, 7, 4, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 3, 1, 1, 0)
		)

	def forward(self, x):
		out = self.main(x).squeeze()
		return out
class MemoryReplay(object):

	def __init__(self,
				 max_size=10000,
				 bs=64,
				 im_size=84,
				 stack=4):

		self.s = np.zeros((max_size, stack+1, im_size, im_size), dtype=np.float32)
		self.r = np.zeros(max_size, dtype=np.float32)
		self.a = np.zeros(max_size, dtype=np.int32)
		#self.ss = np.zeros_like(self.s)
		self.done = np.array([True]*max_size)

		self.max_size = max_size
		self.bs = bs
		self._cursor = None
		self.total_idx = list(range(self.max_size))


	def put(self, sras):

		if self._cursor == (self.max_size-1) or self._cursor is None :
			self._cursor = 0
		else:
			self._cursor += 1

		self.s[self._cursor] = sras[0]
		self.a[self._cursor] = sras[1]
		self.r[self._cursor] = sras[2]
		#self.ss[self._cursor] = sras[3]
		self.done[self._cursor] = sras[3]


	def batch(self):

		sample_idx = random.sample(self.total_idx, self.bs)
		s = self.s[sample_idx, :4]
		a = self.a[sample_idx]
		r = self.r[sample_idx]
		#ss = self.ss[sample_idx]
		ss = self.s[sample_idx, 1:]
		done = self.done[sample_idx]

		return s, a, r, ss, done
MP = MemoryReplay(memory_size, batch_size)
dqn = DQN()
target_dqn = DQN()
target_dqn.load_state_dict(dqn.state_dict())
optimz = optim.RMSprop(dqn.parameters(), lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)

args = parser.parse_args()
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        dqn.load_state_dict(torch.load(args.resume))
        target_dqn.load_state_dict(dqn.state_dict())

dqn.to(device)
target_dqn.to(device)


class Pong(object):

    def __init__(self):
        self.env = gym.make('PongDeterministic-v0')
        self.current_phi = None
        self.reset()

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        obs = self._rbg2gray(obs)
        phi_next = self._phi(obs)

        phi_phi = np.vstack([self.current_phi, obs[np.newaxis]])
        self.current_phi = phi_next

        return phi_phi, r, done

    def reset(self):
        x = self.env.reset()
        x = self._rbg2gray(x)
        phi = np.stack([x, x, x, x])
        self.current_phi = phi

        return phi

    def _rbg2gray(self, x):
        x = x.astype('float32')
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (84, 84)) / 127.5 - 1.

        return x

    def _phi(self, x):
        new_phi = np.zeros((4, 84, 84), dtype=np.float32)
        new_phi[:3] = self.current_phi[1:]
        new_phi[-1] = x

        return new_phi

    def display(self):
        self.env.render()
pong = Pong()

for i in range(memory_size):
    phi = pong.current_phi
    act_index = random.randrange(3)
    phi_next, r, done = pong.step(VALID_ACTION[act_index])
    # pong.display()
    MP.put((phi_next, act_index, r, done))

    if done:
        pong.reset()

print("================\n"
      "Start training!!\n"
      "================")
pong.reset()

epoch = 0
update_count = 0
score = 0.
avg_score = -21.0
best_score = -21.0

t = time.time()

SCORE = []
QVALUE = []
QVALUE_MEAN = []
QVALUE_STD = []

while (epoch < max_epoch):

    while (not done):

        optimz.zero_grad()

        act_index = sample_action(pong, dqn, var_phi, epsilon)

        epsilon = (epsilon - 1e-6) if epsilon > 0.1 else 0.1

        phi_next, r, done = pong.step(VALID_ACTION[act_index])
        MP.put((phi_next, act_index, r, done))
        r = np.clip(r, -1, 1)
        score += r

        # batch sample from memory to train
        batch_phi, batch_a, batch_r, batch_phi_next, batch_done = MP.batch()
        var_batch_phi_next.data.copy_(torch.from_numpy(batch_phi_next))
        batch_target_q, _ = target_dqn(var_batch_phi_next).max(dim=1)

        mask_index = np.ones((batch_size, 1))
        mask_index[batch_done] = 0.0
        var_batch_r_mask.data.copy_(torch.from_numpy(mask_index))
        # print(torch.from_numpy(batch_r).shape)
        var_batch_r.data.copy_(torch.from_numpy(batch_r).view(-1, 1))

        y = var_batch_r + batch_target_q.mul(GAMMA).mul(var_batch_r_mask)
        y = y.detach()

        var_batch_phi.data.copy_(torch.from_numpy(batch_phi))
        batch_q = dqn(var_batch_phi)

        var_batch_a.data.copy_(torch.from_numpy(batch_a).long().view(-1, 1))
        batch_q = batch_q.gather(1, var_batch_a)

        loss = y.sub(batch_q).pow(2).mean()
        loss.backward()
        optimz.step()

        update_count += 1

        if update_count == update_step:
            target_dqn.load_state_dict(dqn.state_dict())
            update_count = 0

        QVALUE.append(batch_q.data.cpu().numpy().mean())

    SCORE.append(score)
    QVALUE_MEAN.append(np.mean(QVALUE))
    QVALUE_STD.append(np.std(QVALUE))
    QVALUE = []

    save_statistic('Score', SCORE, save_path=save_path)
    save_statistic('Average Action Value', QVALUE_MEAN, QVALUE_STD, save_path)

    pong.reset()
    done = False
    epoch += 1
    avg_score = 0.9 * avg_score + 0.1 * score
    score = 0.0
    print('Epoch: {0}. Avg.Score:{1:6f}'.format(epoch, avg_score))

    time_elapse = time.time() - t

    if avg_score >= best_score and time_elapse > 300:
        torch.save(dqn.state_dict(), save_path + '/model.pth')
        print('Model has been saved.')
        best_score = avg_score
        t = time.time()