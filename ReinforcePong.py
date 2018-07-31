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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torch.distributions import Categorical
from pathlib import Path
from PIL import Image
from itertools import count
import seaborn as sns
import skimage
from skimage.color import rgb2gray
import gym
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Turn interactive plotting off
plt.ioff()

Transition = namedtuple('Transition',
                            ('state', 'action', 'reward'))
# PATH = "/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/"
PATH = "C:\\Users\\Joseph\\PycharmProjects\\SURE\\"
BATCH_SIZE = 10000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TARGET_UPDATE = 10
DISCOUNT_RATE = 0.95
MEMORY_SIZE = 100000
TRAINING_LENGTH = 10000
OBSERVATION_SIZE = 80 * 80
steps_done = 0
scoreSaveLength = 0
durationSaveLength = 0
VALID_ACTION = [0, 2, 3]
env = gym.make('PongDeterministic-v0')
actions = []
losses = []
parser = argparse.ArgumentParser(description='QLearningPong')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument("--lr", default=0.0005, type=float, metavar='LR', help='learning rate')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.head = nn.Linear(1120, 3)

        # self.main = nn.Sequential(
        #     nn.Conv2d(3, 32, 8, 4, 0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 4, 2, 0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 512, 7, 4, 0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 3, 1, 1, 0),
        #     nn.Linear()
        # )

        self.linear1 = nn.Linear(6400, 200)
        self.linear2 = nn.Linear(200, 3)
        self.saved_log_probs = []
        self.rewards = []
    def forward(self, x):
        x = x.float()
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
def main():
    global args
    global actions
    global steps_done
    global losses
    args = parser.parse_args()
    # if gpu is to be used
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    # optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    episode_durations = []
    scores = []
    memory = ReplayMemory(MEMORY_SIZE)
    if args.lr:
        learning_rate = args.lr
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            policy_net.load_state_dict(checkpoint['state_dict'])
            target_net.load_state_dict(checkpoint['target_dict'])
            optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate, alpha=0.99, eps=1e-10, momentum=0.0)
            optimizer.load_state_dict(checkpoint['optimizer'])
            episode_durations = checkpoint['episodes']
            scores = checkpoint['scores']
            steps_done = checkpoint['steps_done']
            losses = checkpoint['losses']
            target_net.eval()
    else:
        optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate, alpha=0.99, eps=1e-10, momentum=0.0)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    def preprocess(image):
        # taken from https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/helpers.py
        # prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        image = image[35:195] # crop
        image = image[::2, ::2, 0] # downsize by factor of 2
        image[Image == 144] = 0 # erase background (background type 1)
        image[Image == 109] = 0 # erase background ( background type 2)
        image[image != 0] = 1 # everything else ( ball, paddles) just set to 1
        return torch.from_numpy(image.astype(np.float).ravel()).float().to(device)
    def discount_rewards(rewards, gamma):
        # taken from https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/helpers.py
        rewards = np.array(rewards)
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r.tolist()
    def saveFrame(previousName, currentName, action, reward):
        file = Path("{}/practiceFrames/Transitions.csv".format(PATH))
        if file.exists():
            with open(file, "a") as f:
                f.write("\n{},{},{},{}".format(previousName, currentName, action, reward))
        else:
            file.touch() # create the csv
            with open(file, "a") as f:
                f.write("\n{},{},{},{}".format(previousName, currentName, action, reward))
    def correctPixelArray(nparray):
        return torch.from_numpy(np.ascontiguousarray(np.flip(rgb2gray(nparray).transpose(1, 0), axis=0), dtype=np.uint8)).to(device)
    def optimize_model():
        if len(memory) < BATCH_SIZE: # I'm using Replay Memory and sampling a batch from it randomly
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions)) # this takes the batch, which is a bunch of Transition tuples defined at the beginning of the code
        # and transforms them into a single Transition with all of the respective current_screen, action, next_screen, reward as matrices
        # Compute a mask of non-final states and concatenate the batch elements
        train_observations = torch.cat(batch.state).to(device)
        labels = torch.tensor(batch.action).to(device)
        processed_rewards = torch.cat(batch.reward).to(device)

        # Compute Huber loss
        model_actions = policy_net(train_observations)

        # REINFORCE
        policy_loss = []
        for log_prob, reward in zip(labels.tolist(), processed_rewards.tolist()):
            policy_loss.append(torch.tensor([-log_prob * reward]))
        optimizer.zero_grad()
        policy_loss = Variable(torch.cat(policy_loss).sum(), requires_grad=True)
        policy_loss.backward()
        optimizer.step()
        # # cross_entropies = F.cross_entropy(model_actions, labels.squeeze(), reduce=False)
        # print(cross_entropies)
        # # probs = F.softmax(model_actions)
        # # move_cost = 0.01 * (probs * [0, 1.0, 1.0]).sum()
        # loss = (torch.mul(processed_rewards, cross_entropies.double())).sum()
        # # loss = cross_entropies.sum()
        losses.append(policy_loss)
        # losses.append(loss)
        # Optimize the model


        # loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # optimizer.step()


    def select_action(current_screen):
        global steps_done
        steps_done += 1
        with torch.no_grad():
            # actions.append(policy_net(current_screen).max(1)[1].view(1, 1).data)
            # print(policy_net(current_screen))
            probabilities = F.softmax(policy_net(current_screen))
            m = Categorical(probabilities)
            action = m.sample()
            policy_net.saved_log_probs.append(m.log_prob(action))
            return action.item(), torch.tensor(m.log_prob(action))



    def save_checkpoint(state, filename):
        torch.save(state, filename)

    num_episodes = 100000

    def plot_score():
        global scoreSaveLength
        fig = plt.figure(2)
        plt.clf()
        score_t = torch.tensor(scores, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(score_t.numpy())
        if len(score_t) > scoreSaveLength:  # save plot every 10 episodes
            saveLength = scoreSaveLength + 10
            # plt.savefig("C:\\Users\\Joseph\\PycharmProjects\\SURE\\scoreplt.png")
            plt.savefig(PATH+"reinforcescoreplt.png")
            plt.close(fig)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
    def plot_durations():
        global durationSaveLength
        fig = plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if len(durations_t) > durationSaveLength:  # save plot every 10 episodes
            saveLength = durationSaveLength + 10
            # plt.savefig("C:\\Users\\Joseph\\PycharmProjects\\SURE\\durationsplt.png")
            plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/durationsplt.png")
            plt.close(fig)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def plotLoss():
        plot = plt.figure(num="Loss")
        plt.clf()
        losses_t = torch.tensor(losses, dtype=torch.float)
        plt.title('Loss...')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())
        # plt.savefig("C:\\Users\\Joseph\\PycharmProjects\\SURE\\lossplt.png")
        plt.savefig(PATH + "reinforcelossplt.png")
        plt.close(plot)
    def removeOldestElement(list):
        list.reverse()
        list.pop()
        list.reverse()
    episode_memory = []
    epoch_memory = []
    score = 0
    running_score = 0
    flag = True
    episodes_left = num_episodes - len(scores)
    for i_episode in range(episodes_left):
        print("Episode = " + str(len(scores)))
        current_state = env.reset()
        observation = torch.from_numpy(np.zeros(OBSERVATION_SIZE)).to(device)
        for t in count():

            action, log_prob = select_action(observation)
            next_state, reward, done, info = env.step(VALID_ACTION[action])
            # reward = np.clip(reward, -1, 1)
            score += reward
            # Move to the next state

            episode_memory.append(Transition(observation, log_prob, reward))
            previous_state = current_state
            current_state = next_state
            current_x = preprocess(current_state)
            previous_x = preprocess(previous_state)
            observation = current_x - previous_x

            if done:
                obs, acts, rewards_t = zip(*episode_memory)
                # processed rewards
                processed_rewards = discount_rewards(rewards_t, DISCOUNT_RATE)
                processed_rewards -= np.mean(processed_rewards)
                processed_rewards /= np.std(processed_rewards)
                epoch_memory.extend(list(zip(obs, acts, torch.from_numpy(processed_rewards).to(device))))
                scores.append(score)
                running_score = running_score * 0.99 + score*0.01
                if(running_score >= 19):
                    #then we have solved it, exit
                    exit(0)
                plot_score()
                plotLoss()
                if(flag == True):
                    save_checkpoint({'episodes': episode_durations,
                                     'state_dict': policy_net.state_dict(),
                                     'target_dict': target_net.state_dict(),
                                     'optimizer': optimizer.state_dict(), 'scores': scores, 'steps_done': steps_done,
                                     'losses': losses}, PATH + "reinforcecheckpoint1.pt")
                    flag = not flag
                else:
                    save_checkpoint({'episodes': episode_durations,
                                     'state_dict': policy_net.state_dict(),
                                     'target_dict': target_net.state_dict(),
                                     'optimizer': optimizer.state_dict(), 'scores': scores, 'steps_done': steps_done,
                                     'losses': losses}, PATH + "reinforcecheckpoint2.pt")
                    flag = not flag
                score = 0
                episode_memory.clear()
                env.reset()
                if(len(epoch_memory) >= TRAINING_LENGTH):
                    print("Running Score: ", running_score)
                    break
                # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        for i in list(zip(*zip(*epoch_memory))):
            memory.push(i[0].unsqueeze(0).float(), i[1], i[2].unsqueeze(0))
        epoch_memory.clear()
        optimize_model()
if __name__ == "__main__":
    main()