from builtins import range
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
from torch.autograd import Variable
from torch.utils.data import dataset, DataLoader
from pathlib import Path
from PIL import Image
from itertools import count
PATH = "practiceFrames/"
Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 4
GAMMA = 0.9
losses = []


def correctPixelArray(nparray):
    return np.ascontiguousarray(np.flip(nparray.transpose(2, 1, 0), axis=0), dtype=np.float32)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(1120, 3)

    def forward(self, x):
        x = x.float() / 256
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
class MyDataSet(dataset.Dataset):
    def __init__(self, imageDir, csvPath):
        super(MyDataSet, self).__init__()
        self.frame = pd.read_csv(csvPath)
        self.dir = imageDir

    def __len__(self):
        return len(self.frame)
    def __getitem__(self, item):
        actionList = []
        rewardList = []
        previousImageName = os.path.join(self.dir, self.frame.iloc[item, 0])
        previousImage = np.array(Image.open(previousImageName))
        currentImageName = os.path.join(self.dir, self.frame.iloc[item, 1])
        currentImage = np.array(Image.open(currentImageName))
        action = self.frame.iloc[item, 2]
        action = action.tolist()
        actionList.append(action)
        actionList = [torch.tensor(a, dtype = torch.long) for a in actionList]
        reward = self.frame.iloc[item, 3]
        reward = reward.tolist()
        rewardList.append(reward)
        rewardList = [torch.tensor(r, dtype = torch.float32) for r in rewardList]
        return Transition(correctPixelArray(previousImage), torch.tensor(actionList), correctPixelArray(currentImage), torch.tensor(rewardList))
def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = CNN().to(device)
    data = MyDataSet(PATH, os.path.join(PATH, "Transitions.csv"))
    loader = DataLoader(dataset=data, batch_size=BATCH_SIZE, num_workers=1)
    optimizer = optim.RMSprop(model.parameters())


    def train(model):
        for sample_batched in loader:
            # detailed explanation).
            sample_batch = Transition(sample_batched[0], sample_batched[1], sample_batched[2], sample_batched[3])
            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    sample_batch.next_state)), device=device, dtype=torch.uint8)
            non_final_next_states = sample_batch.next_state
            state_batch = sample_batch.state
            action_batch = sample_batch.action
            reward_batch = torch.cat(tuple(sample_batch.reward))
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = model(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            print(action_batch.shape)
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            losses.append(loss)
            # Optimize the model
            optimizer.zero_grad()

            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
    def plotLoss():
        plot = plt.figure(num="Loss")
        plt.clf()
        losses_t = torch.tensor(losses, dtype=torch.float)
        plt.title('Loss...')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.plot(losses_t.numpy())
        plt.savefig("C:\\Users\\Joseph\\PycharmProjects\\SURE\\lossplt.png")
        # plt.savefig("/home/josephp/projects/def-dnowrouz/josephp/Pong/lossplt.png")
        plt.close(plot)
    train(model)
    plotLoss()
if __name__ == "__main__":
        main()