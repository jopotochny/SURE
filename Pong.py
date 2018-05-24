# Taken from: https://github.com/hamdyaea/Daylight-Pong-python3
# Modified by Joseph Potochny

import os
import random
from collections import namedtuple
import pygame
import sys
from pygame import *
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


window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Daylight Pong')

# image file base name
# new_images_path = "C:\\Users\\Joseph\\PycharmProjects\\SURE\\dataset\\"
# old_images_path = "C:\\Users\\Joseph\\PycharmProjects\\Pytorch_Data_Loading_Tutorial\\faces"
# image_base_name = "image_"
# file_extension = ".jpg"
# initial_file_count = len([f for f in os.listdir(new_images_path)])
# do some preliminary reinforcement learning stuff- teach to beat pong, etc?
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
    if event.key == K_UP:
        rightInput = "K_UP"
        paddle2_vel = -8
    elif event.key == K_DOWN:
        rightInput = "K_DOWN"
        paddle2_vel = 8
    elif event.key == K_s:
        leftInput = "K_s"
        paddle1_vel = -8
    elif event.key == K_z:
        leftInput = "K_z"
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel
    global leftInput
    global rightInput
    if event.key in (K_z, K_s):
        leftInput = "None"
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        rightInput = "None"
        paddle2_vel = 0


init()
randomGenerator = random.Random(217)
computer1 = RandomComputer.RandomComputer(1, randomGenerator)
randomGenerator2 = random.Random(5673)
computer2 = RandomComputer.RandomComputer(2, randomGenerator2)
flag = True
while True:

    draw(window)

    #computer1.update(ball_pos, paddle1_pos)
    #computer2.update(ball_pos, paddle2_pos)

    computer1.update()
    computer2.update()

    for event in pygame.event.get():

        if event.type == KEYDOWN:
            keydown(event)
        elif event.type == KEYUP:
            keyup(event)
        elif event.type == QUIT:
            pygame.quit()
            sys.exit()
    # if(flag):
    #     previousFrameName = new_images_path + image_base_name + str(initial_file_count) + file_extension
    #     initial_file_count += 1
    #     pygame.image.save(window, previousFrameName)
    print(torch.from_numpy(np.flip(pygame.surfarray.pixels3d(window).transpose(2, 1, 0), axis=0).copy()).size())
    pygame.display.update()
    #     nextFrameName = new_images_path + image_base_name + str(initial_file_count) + file_extension
    #     pygame.image.save(window, nextFrameName)
    #     dataTuple = (leftInput, rightInput, previousFrameName, nextFrameName)
    #     csvFile = open(new_images_path+"controls.csv", "w")
    #     csvFile.write(str(dataTuple) + "\n")
    #     initial_file_count += 1
    #     flag = False
    fps.tick(60)