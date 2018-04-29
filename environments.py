import copy as cp
import random
import numpy as np
import turtle
import time
import os,sys
# import CNN
random.seed(1234)
np.random.seed(1234)


class State():
    def next_state(self,action):
        raise NotImplementedError()

    def reward(self,action,next_state):
        raise NotImplementedError()


def RiverSwimEnv():
    return 0


class RiverSwimState(State):

    def __init__(self,obs=0):
        # super(RiverSwimState,self).__init__()
        self.state = obs
        self.number_of_visits = 1
        self.actions = {'L':{'nov':0,'Q':0},'R':{'nov':0,'Q':0}}
        self.eps    = 0.1
        self.untried_actions = {}
        self.term = False
        self.policy = 0
        self.n_states = 5

    def get_actions(self,state):
        return ['L','R']

    def next_state_reward(self,action):
        if action == 'L':
            if self.state == 0:
                next = RiverSwimState(np.random.choice([0, 1], p=[0.9, 0.1]))
            elif self.state == self.n_states - 1:
                next = RiverSwimState(np.random.choice([self.state, self.state - 1], p=[self.eps, 1.0 - self.eps]))
            else:
                next = RiverSwimState(np.random.choice([self.state - 1, self.state, self.state + 1],
                                                  p=[1.0 - self.eps, self.eps / 2, self.eps / 2]))
        # right
        elif self.state == self.n_states - 1:
            next = RiverSwimState(np.random.choice([self.state, self.state - 1], p=[1.0 - self.eps, self.eps]))
        elif self.state > 0:
            next = RiverSwimState(np.random.choice([self.state + 1, self.state, self.state - 1],
                                              p=[0.5 - self.eps / 2, 0.5 - self.eps / 2, self.eps]))
        else:
            next = RiverSwimState(np.random.choice([0, 1], p=[0.55, 0.45]))


        if self.state == 4:
            next.term = 1
        #reward calc
        if self.state==0 or (self.state==1 and action=='L'):
            reward = 0.0001
        elif self.state==self.n_states-1 or (self.state==self.n_states-2 and action=='R'):
            reward = 10.0000
        else:
            reward = 0.0
        return next,reward

    def env_reset(self):
        return RiverSwimState()



class Canvas:

    def __init__(self,obs=turtle.getscreen()):
        self.state = obs
        self.nspeed = 2
        self.smin = 0
        self.smax = 2
        self.nrot = 7
        self.rmax = 30
        self.rinc = self.rmax*2/self.nrot
        self.n_act = self.nspeed * self.nrot
        self.actions ={}
        speed = range(self.smin,self.smax)
        rot = np.arange(-self.rmax,self.rmax,self.rinc)
        for i in speed:
            for j in rot:
                self.actions.update({(i,j):{'nov':0,'Q':0}})
        self.term = False
        turtle.penup()

    def get_actions(self,state):
        speed = range(self.smin,self.smax)
        rot = np.arange(-self.rmax,self.rmax,self.rinc)
        for i in speed:
            for j in rot:
                self.actions.update({(i,j):{'nov':0,'Q':0}})
        return self.actions

    def next_state(self,action):
        # turtle.tracer(0, 0)
        turtle.speed(action[0])
        turtle.pendown()
        turtle.right(action[1])
        turtle.forward(10)
        return Canvas(turtle.getscreen())
        #draw on the canvas everytime this is called

        
    def reward(self):
        #generate an image and pass to CNN to get reward
        # turtle.update()
        # os.system("rm image.bmp")
        ts = turtle.getscreen()
        ts.getcanvas().postscript(file="image.eps")
        # reward = CNN.predict(img)
        return 1

    def env_reset(self,path):
        turtle.reset()
        # draw the current best actions
        return 1
