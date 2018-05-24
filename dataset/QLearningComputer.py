import pygame
import random
from pygame import *
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class QLearningComputer:
    downKey = 0
    upKey = 0
    player = 0

    def updateR(self, ball_position, paddle_postition):
        if(ball_position[1] > paddle_postition[1]): # then need to update re




