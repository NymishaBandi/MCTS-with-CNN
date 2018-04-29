import turtle, random, time

turtle.title('Test')
turtle.shape('turtle')
turtle.color('white')
turtle.fillcolor('white')
turtle.bgcolor('skyblue')
turtle.pensize(20)
turtle.speed(100000)
turtle.penup()



def semicircle(size):
    turtle.pendown()
    for i in range(0,20):
        turtle.forward(size)
        turtle.left(9)
    turtle.left(180)

def adjustturn():
    turtle.left(30)

for i in range(0,25):
    semicircle(5)
    adjustturn()