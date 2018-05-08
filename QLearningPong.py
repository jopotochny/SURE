# Taken from: https://github.com/hamdyaea/Daylight-Pong-python3
# Modified by Joseph Potochny

import os
import random
from collections import namedtuple
import pygame
import math
import sys
import matplotlib
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
import RandomComputer
pygame.init()
fps = pygame.time.Clock()

Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=5)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(384, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
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
# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(500)

def optimize_model():
    # print("Optimizing")
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
    state_batch = torch.cat(batch.state) # 128 images of size (1, 3, 40, 80)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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

steps_done = 0


def select_action(state):
    # print("Selecting Action")
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

WHITE = (255, 255, 255)
ORANGE = (255,140,0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0, 0]
ball_vel = [0, 0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0
last_l_score = 0
last_r_score = 0

def calculateReward(player):
    # print("Calculating Reward")
    global l_score, r_score, last_l_score, last_r_score
    if(player == 1): # then we are left paddle
        if(last_l_score < l_score): # then we scored a point since we last checked
            last_l_score = l_score
            return torch.tensor([1], dtype=torch.float)
        elif(last_r_score < r_score): # then we have been scored on
            last_r_score = r_score
            return torch.tensor([-1], dtype=torch.float)
        else: # score hasn't changed
            return torch.tensor([0], dtype=torch.float)
    else:
        if (last_r_score < r_score):  # then we scored a point since we last checked
            last_r_score = r_score
            return torch.tensor([1], dtype=torch.float)
        elif (last_l_score < l_score):  # then we have been scored on
            last_l_score = l_score
            return torch.tensor([-1], dtype=torch.float)
        else:  # score hasn't changed
            return torch.tensor([0], dtype=torch.float)

def performAction(action, player):
    if(player == 1): # then we are left paddle
        up = pygame.K_z
        down = pygame.K_s
        if(action == 0): # then we move up
            new_event = pygame.event.Event(pygame.KEYDOWN, {up: up})
            new_event.key = up
            pygame.event.post(new_event)
        else: # action = 1 and we move down
            new_event = pygame.event.Event(pygame.KEYDOWN, {down: down})
            new_event.key = down
            pygame.event.post(new_event)
    else: # we are right paddle
        up = pygame.K_UP
        down = pygame.K_DOWN
        if (action == 0):  # then we move up
            new_event = pygame.event.Event(pygame.KEYDOWN, {up: up})
            new_event.key = up
            pygame.event.post(new_event)
        else:  # action = 1 and we move down
            new_event = pygame.event.Event(pygame.KEYDOWN, {down: down})
            new_event.key = down
            pygame.event.post(new_event)

def isDone():
    global l_score, r_score
    if(l_score >= 10 or r_score >= 10):
        return True
    else:
        return False

window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Daylight Pong')

# image file base name
new_images_path = "C:\\Users\\Joseph\\PycharmProjects\\SURE\\dataset\\"
old_images_path = "C:\\Users\\Joseph\\PycharmProjects\\Pytorch_Data_Loading_Tutorial\\faces"
image_base_name = "image_"
file_extension = ".jpg"
initial_file_count = len([f for f in os.listdir(new_images_path)])
leftInput = 0
rightInput = 0
def ball_init(right):
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    horz = random.randrange(2, 4)
    vert = random.randrange(1, 3)

    if right == False:
        horz = - horz

    ball_vel = [horz, -vert]


def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel, l_score, r_score  # these are floats
    global score1, score2  # these are ints
    global computer
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
    paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT //2]
    l_score = 0
    r_score = 0
    if random.randrange(0, 2) == 0:
        ball_init(True)
    else:
        ball_init(False)


def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score

    canvas.fill(BLACK)
    pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)


    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel

    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel


    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])


    pygame.draw.circle(canvas, ORANGE, ball_pos, 20, 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)


    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = - ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]


    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(paddle1_pos[1] - HALF_PAD_HEIGHT,
                                                                                 paddle1_pos[1] + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)

    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(
            paddle2_pos[1] - HALF_PAD_HEIGHT, paddle2_pos[1] + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)


    myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
    label1 = myfont1.render("Score " + str(l_score), 1, (255, 255, 0))
    canvas.blit(label1, (50, 20))

    myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
    label2 = myfont2.render("Score " + str(r_score), 1, (255, 255, 0))
    canvas.blit(label2, (470, 20))


def keydown(event):
    global paddle1_vel, paddle2_vel
    global initial_file_count
    global leftInput
    global rightInput
    #landmarks_frame = pd.read_csv(new_images_path + 'controls.csv')
    if event.key == pygame.K_UP:
        rightInput = "K_UP"
        paddle2_vel = -8
    elif event.key == pygame.K_DOWN:
        rightInput = "K_DOWN"
        paddle2_vel = 8
    elif event.key == pygame.K_s:
        leftInput = "K_s"
        paddle1_vel = -8
    elif event.key == pygame.K_z:
        leftInput = "K_z"
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel
    global leftInput
    global rightInput
    if event.key in (pygame.K_z, pygame.K_s):
        leftInput = "None"
        paddle1_vel = 0
    elif event.key in (pygame.K_UP, pygame.K_DOWN):
        rightInput = "None"
        paddle2_vel = 0


init()
randomGenerator = random.Random(217)
computer1 = RandomComputer.RandomComputer(1, randomGenerator)
randomGenerator2 = random.Random(5673)
computer2 = RandomComputer.RandomComputer(2, randomGenerator2)
flag = True
num_episodes = 300

episode_durations = []

def plot_durations():
    plt.figure(2)
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
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



for i_episode in range(num_episodes):
    draw(window)
    # computer1.update(ball_pos, paddle1_pos)
    # computer2.update(ball_pos, paddle2_pos)
    window.lock()
    last_screen = np.ascontiguousarray(np.flip(pygame.surfarray.pixels3d(window).transpose(2, 1, 0), axis=0).copy(), dtype=np.float32)
    window.unlock()
    last_screen = torch.from_numpy(last_screen).unsqueeze(0).to(device)
    current_screen = last_screen
    state = current_screen - last_screen
    for t in count():
        draw(window)
        # Select and perform an action
        action = select_action(state)
        reward = calculateReward(player=1).item()
        reward = torch.tensor([reward], device=device)
        performAction(action, player=1)
        computer2.update()
        # Observe new state
        last_screen = current_screen
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                keydown(event)
            elif event.type == pygame.KEYUP:
                keyup(event)
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()
        fps.tick(60)
        window.lock()
        current_screen = np.ascontiguousarray(np.flip(pygame.surfarray.pixels3d(window).transpose(2, 1, 0), axis=0).copy(), dtype=np.float32)
        window.unlock()
        current_screen = torch.from_numpy(current_screen).unsqueeze(0).to(device)
        done = isDone()
        if not done:
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
            plot_durations()
            break
        # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


    # if(flag):
    #     previousFrameName = new_images_path + image_base_name + str(initial_file_count) + file_extension
    #     initial_file_count += 1
    #     pygame.image.save(window, previousFrameName)
    #     nextFrameName = new_images_path + image_base_name + str(initial_file_count) + file_extension
    #     pygame.image.save(window, nextFrameName)
    #     dataTuple = (leftInput, rightInput, previousFrameName, nextFrameName)
    #     csvFile = open(new_images_path+"controls.csv", "w")
    #     csvFile.write(str(dataTuple) + "\n")
    #     initial_file_count += 1
    #     flag = False
