
# this is the definition of the state class

import copy as cp
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
import time
import math
from  environments import Canvas as TaskState
import  environments as env

       



def MCTS(state,d):
    for i in range(budget):
        TaskState.env_reset(state,path)
        SIMULATE(state,d)
        # if i % 100==0:
        #     print i
    # for k in state.actions.keys():
        # print 'Q(%d,%s) = %f\n' % (state.state,str(k),state.actions[k]['Q'])
        # print (str(k),state.actions[k]['Q'])
    return argmax(state)


def SIMULATE(s,d):
    if d==0 or s.term == True:
        return reward(model)
    if s not in T:
        T.append(s)
        s.number_of_visits = 1
        return ROLLOUT(s,d)
    s.number_of_visits += 1
    if untried_actions(s):
        a = random.choice(untried_actions(s))
        # s.untried_actions.pop(a)
    else:
        a = argmax(s,1)
    next_state = TaskState.next_state(s,a)
    q = SIMULATE(next_state,d-1)
    s.actions[a]['nov']+=1
    s.actions[a]['Q'] = s.actions[a]['Q'] + (q-s.actions[a]['Q'])/s.actions[a]['nov']
    return q

def ROLLOUT(s,d):
    if d==0 or s.term == True:
        return TaskState.reward(s,model)
    else:
        a = random.choice(list(s.actions.keys()))
        next_state = TaskState.next_state(s,a)
    return ROLLOUT(next_state,d-1)
 
def argmax(state,cons=0):
    maximum = -100000
    max_action = []
    for k in state.actions.keys():
        # print 'Q(%d,%d) = %f\n' % (state.state,k,state.actions[k]['Q'])
        if cons == 1:
            constant = c * math.sqrt(math.log(state.number_of_visits / state.actions[k]['nov']))
            if state.actions[k]['Q']+constant >= maximum:
                maximum = state.actions[k]['Q']+constant
                max_action.append(k)
        elif state.actions[k]['Q'] > maximum:
            maximum = state.actions[k]['Q']
            max_action = [k]
        elif state.actions[k]['Q'] == maximum:
            max_action.append(k)
    return random.choice(max_action)

def untried_actions(s):
    untried_actions=[]
    for k in s.actions.keys():
        if s.actions[k]['nov']==0:
            untried_actions.append(k)
    return untried_actions






if __name__=='__main__':


    T = []
    model= env.model()
    d       = 50    #depth to which the MCTS will explore
    budget  =100    #computational budget
    gamma   =  1     #discount
    c       =  1/math.sqrt(2.0)     #parameter controlling amount of exploration
    path=[]
    root = TaskState()
    for i in range(50):
        path.append(list(MCTS(root,d)))

