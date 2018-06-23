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
from pathlib import Path
from PIL import Image
from itertools import count
import seaborn as sns
import gym
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Turn interactive plotting off
plt.ioff()

Transition = namedtuple('Transition',
                            ('current_screen', 'action', 'next_screen', 'reward'))
PATH = "/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/"
# PATH = "C:\\Users\\Joseph\\PycharmProjects\\SURE\\"
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TARGET_UPDATE = 10
steps_done = 0
scoreSaveLength = 0
durationSaveLength = 0
VALID_ACTION = [0, 3, 4]
env = gym.make('PongDeterministic-v0')
actions = []
losses = []
parser = argparse.ArgumentParser(description='QLearningPong')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to latest checkpoint')
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
        self.conv1 = nn.Conv2d(3, 16, 8, 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(32, 64, 3, 2)
        # self.bn4 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(22528, 3)
        # self.linear2 = nn.Linear(512, 3)

    def forward(self, x):
        x = x.float() / 256
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = self.linear1(x.view(x.size(0), -1))
        return x
        # out = self.main(x).squeeze()
        # print(out.shape)
        # return out
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
    args = parser.parse_args()
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    optimizer = optim.RMSprop(policy_net.parameters(),  lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)
    # optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    episode_durations = []
    scores = []
    memory = ReplayMemory(2000)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            policy_net.load_state_dict(checkpoint['state_dict'])
            target_net.load_state_dict(policy_net.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            episode_durations = checkpoint['episodes']
            scores = checkpoint['scores']
            target_net.eval()
    else:
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

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
        return np.ascontiguousarray(np.flip(nparray.transpose(2, 1, 0), axis=0), dtype=np.float32)
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_screen)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_screen
                                           if s is not None])
        state_batch = torch.cat(batch.current_screen).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        losses.append(loss)
        # Optimize the model
        optimizer.zero_grad()

        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    def select_action(current_screen):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # actions.append(policy_net(current_screen).max(1)[1].view(1, 1).data)
                # print(policy_net(current_screen))
                return policy_net(current_screen).max(1)[1].view(1, 1)
        else:
            randomAction = torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)
            actions.append(randomAction)
            return randomAction


    def save_checkpoint(state, filename):
        torch.save(state, filename)

    num_episodes = 5000

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
            plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/scoreplt.png")
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
        plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/SURE/lossplt.png")
        plt.close(plot)
    saveLimit = 1000
    saveFlag = 0
    score = 0
    episodes_left = num_episodes - len(episode_durations)
    for i_episode in range(episodes_left):
        print("Episode = " + str(len(episode_durations)))
        current_screen = env.reset()
        current_screen = correctPixelArray(current_screen)
        current_screen = torch.from_numpy(current_screen).unsqueeze(0).to(device)
        for t in count():

            action = select_action(current_screen)
            next_screen, reward, done, info = env.step(VALID_ACTION[action])
            next_screen = correctPixelArray(next_screen)
            next_screen = torch.from_numpy(next_screen).unsqueeze(0).to(device)
            reward = np.clip(reward, -1, 1)
            score += reward
            # Move to the next state
            memory.push(current_screen, action, next_screen, torch.tensor([reward]))
            # Perform one step of the optimization (on the target network)
            # Store the transition in memory
            optimize_model()
            current_screen = next_screen

            if done:
                episode_durations.append(t + 1)
                scores.append(score)
                plot_score()
                plot_durations()
                plotLoss()
                # plot = sns.distplot(actions)
                # fig = plot.get_figure()
                # fig.savefig("{}/{}".format(PATH, str(len(episode_durations)) + "actions_histogram.png"))
                # actions.clear()# reset the actions buffer
                save_checkpoint({'episodes': episode_durations,
                                     'state_dict': policy_net.state_dict(),
                                     'optimizer': optimizer.state_dict(), 'scores': scores}, PATH + "checkpoint.pt")
                score = 0
                break
                # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    main()