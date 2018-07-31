import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
# PATH = "/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/"
PATH = "C:\\Users\\Joseph\\PycharmProjects\\SURE\\"
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
losses = []

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
TARGET_UPDATE = 100

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []
        self.next_states = []
        self.dones = []

    def forward(self, x):
        x = x.float()
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
target = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def plotLoss():
    plot = plt.figure(num="Loss")
    plt.clf()
    losses_t = torch.tensor(losses, dtype=torch.float)
    plt.title('Loss...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(losses_t.numpy())
    # plt.savefig("C:\\Users\\Joseph\\PycharmProjects\\SURE\\lossplt.png")
    plt.savefig(PATH + "cartpolelossplt.png")
    plt.close(plot)
def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    losses.append(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
def finish_episode_Q_func(gamma):
    states = torch.tensor(policy.states)
    next_states = torch.tensor(policy.next_states)
    rewards = torch.tensor(policy.rewards)
    state_actions = policy(states.unsqueeze(0)).detach().max(2)[0]
    next_actions = target(next_states.unsqueeze(0)).detach().max(2)[0]
    Q_target = rewards + gamma * next_actions * (1 - torch.tensor(policy.dones).float())

    loss = F.mse_loss(state_actions, Q_target)
    var_loss = torch.autograd.Variable(loss, requires_grad=True)
    optimizer.zero_grad()
    var_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.states[:]
    del policy.next_states[:]
    del policy.dones[:]

def main():
    c = 0
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            policy.states.append(state)
            policy.next_states.append(next_state)
            policy.dones.append(done)
            state = next_state
            if done:
                c += 1
                break
        if c % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())
        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
    plotLoss()
