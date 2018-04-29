import turtle, random, time

turtle.title('Test')
turtle.shape('turtle')
turtle.color('white')
turtle.fillcolor('white')
turtle.bgcolor('skyblue')
turtle.pensize(2)
turtle.speed(100000)
turtle.penup()




nspeed = 2
smin = 0
smax = 2
nrot = 7
rmax = 30
rinc = rmax*2/nrot
n_act = nspeed * nrot
actions ={}

import numpy as np
import random
speed = range(smin,smax)
rot = np.arange(-rmax,rmax,rinc)
for i in speed:
    for j in rot:
        actions.update({(i,j):{'nov':0,'Q':0}})

untried_actions=[]
for k in actions.keys():
    if actions[k]['nov']==0:
        untried_actions.append(k)

action = random.choice(untried_actions)
s=action[0]
turtle.speed(s)
turtle.pendown()
turtle.right(action[1])
time.sleep(5)
turtle.forward(10)
time.sleep(5)
# root = tk.Tk()

# label = tk.Canvas(root,width=500,height=500)
# i=label.create_oval(10,10,25,25,fill='red')
# label.move(i,245,100)
# label.pack()

# root.mainloop()


