import pygame
import random
from pygame import *
import secrets

class RandomComputer:
    downKey = 0
    upKey = 0
    timeReset = 3
    player = 0
    timer = timeReset # number of loop iterations before changing behavior
    flags = [False, False, False] # 0th = up, 1st = down, 2nd = stop
    def __init__(self, paddleNum, instanceOfRandom): # paddleNum determines what player the computer is controlling, 1 == left, else right
        self.randomClass = instanceOfRandom
        if(paddleNum == 1):
            self.player = 1
            self.downKey = K_z
            self.upKey = K_s

        else:
            self.player = 2
            self.downKey = K_DOWN
            self.upKey = K_UP

    def update(self):
        if(not(self.flags[0] or self.flags[1] or self.flags[2]) or self.timer == 0): # then flags not set or timer ran out
            randNum1 = self.randomClass.choice([1, 2, 3, 4, 5, 6, 7, 8])
            randNum2 = self.randomClass.choice([1, 2, 3, 4, 5, 6, 7, 8])
            if(self.player == 1):
                if (randNum1 == 1 or randNum1 == 2 or randNum1 == 6):
                    self.flags[0] = False
                    self.flags[1] = True
                    self.flags[2] = False
                elif (randNum1 == 3 or randNum1 == 4 or randNum1 == 5 ):
                    self.flags[0] = True
                    self.flags[1] = False
                    self.flags[2] = False
                else:
                    self.flags[0] = False
                    self.flags[1] = False
                    self.flags[2] = True
                if (self.timer == 0):
                    self.timer = self.timeReset  # reset timer
            else: # if player 2 use second random choice
                if (randNum2 == 1 or randNum2 == 2 or randNum2 == 3 ):
                    self.flags[0] = False
                    self.flags[1] = True
                    self.flags[2] = False
                elif (randNum2 == 5 or randNum2 == 6 or randNum2 == 4):
                    self.flags[0] = True
                    self.flags[1] = False
                    self.flags[2] = False
                else:
                    self.flags[0] = False
                    self.flags[1] = False
                    self.flags[2] = True
                if (self.timer == 0):
                    self.timer = self.timeReset  # reset timer
        if(self.flags[0]):
            new_event = pygame.event.Event(KEYDOWN, {self.downKey: self.downKey})
            new_event.key = self.downKey
            event.post(new_event)
            self.timer -= 1
        elif(self.flags[1]):
            new_event = pygame.event.Event(KEYDOWN, {self.upKey: self.upKey})
            new_event.key = self.upKey
            event.post(new_event)
            self.timer -= 1
        elif(self.flags[2]):
            new_event = pygame.event.Event(KEYUP, {self.downKey: self.downKey})
            new_event.key = self.downKey
            event.post(new_event)
            self.timer -= 1

