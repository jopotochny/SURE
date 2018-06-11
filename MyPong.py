from ple.games.pong import Pong
from ple import PLE
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
from itertools import count
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Turn interactive plotting off
plt.ioff()

Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
PATH = "/home/josephp/projects/def-dnowrouz/josephp/Pong/"
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50
steps_done = 0
saveLength = 0
game = Pong()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()
ACTION_SET = p.getActionSet()

parser = argparse.ArgumentParser(description='QLearningPong')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to latest checkpoint')
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.head = nn.Linear(1024, 3)
        # self.head = nn.DataParallel(self.head)
        # self.conv1 = nn.DataParallel(self.conv1)
        # self.bn3 = nn.DataParallel(self.bn3)
        # self.bn1 = nn.DataParallel(self.bn1)
        # self.conv2 = nn.DataParallel(self.conv2)
        # self.bn2 = nn.DataParallel(self.bn2)
        # self.conv3 = nn.DataParallel(self.conv3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return self.head(x.view(x.size(0), -1))

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
    args = parser.parse_args()
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    episode_durations = []
    scores = []
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

    memory = ReplayMemory(10000)

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
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state).to(device)
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
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)


    def save_checkpoint(state, filename):
        torch.save(state, filename)

    num_episodes = 1000

    def plot_score():
        global saveLength
        fig = plt.figure(2)
        plt.clf()
        score_t = torch.tensor(scores, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(score_t.numpy())
        if len(score_t) > saveLength:  # save plot every 10 episodes
            saveLength = saveLength + 10
            plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/plt.png")
            plt.close(fig)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
    def plot_durations():
        global saveLength
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
        if len(durations_t) > saveLength:  # save plot every 50 episodes
            saveLength = saveLength + 10
            plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/plt.png")
            plt.close(fig)
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    episodes_left = num_episodes - len(episode_durations)
    for i_episode in range(episodes_left):
        print("Episode = " + str(len(episode_durations)))
        last_screen = correctPixelArray(p.getScreenRGB())
        last_screen = torch.from_numpy(last_screen).unsqueeze(0).to(device)
        current_screen = last_screen
        state = current_screen - last_screen
        for t in count():
            action = select_action(state)
            reward = torch.Tensor([p.act(ACTION_SET[action])])
            last_screen = current_screen
            current_screen = correctPixelArray(p.getScreenRGB())
            current_screen = torch.from_numpy(current_screen).unsqueeze(0).to(device)
            done = p.game_over()
            if not done:  # check if the game is over
                next_state = current_screen - last_screen
            else:
                next_state = None
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            optimize_model()

            if done:

                episode_durations.append(t + 1)
                scores.append(p.score())
                plot_score()
                # plot_durations()
                save_checkpoint({'episodes': episode_durations,
                                     'state_dict': target_net.state_dict(),
                                     'optimizer': optimizer.state_dict(), 'scores': scores}, PATH + "checkpoint.pt")
                p.reset_game()
                break
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    main()