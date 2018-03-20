import random
import pygame
import sys
from pygame import *

class Computer:
    downKey = 0
    upKey = 0
    def __init__(self, paddleNum):
        if(paddleNum == 1):
            self.downKey = K_z
            self.upKey = K_s
        else:
            self.downKey = K_DOWN
            self.upKey = K_UP
    def update(self, ball_position, paddle_position):
        if(ball_position[1] > paddle_position[1]):
            new_event = pygame.event.Event(KEYDOWN, {self.downKey: self.downKey })
            new_event.key = self.downKey
            event.post(new_event)
        elif(ball_position[1] < paddle_position[1]):
            new_event = pygame.event.Event(KEYDOWN, {self.upKey: self.upKey})
            new_event.key = self.upKey
            event.post(new_event)
        elif (ball_position[1] == paddle_position[1]):
            new_event = pygame.event.Event(KEYUP, {self.downKey: self.downKey})
            new_event.key = self.downKey
            event.post(new_event)
