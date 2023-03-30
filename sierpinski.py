# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:28:26 2023

@author: 20pt20
"""

import turtle

def sierpinski(length, depth):
    if depth == 0:
        for _ in range(3):
            turtle.forward(length)
            turtle.left(120)
    else:
        new_length = length / 2
        sierpinski(new_length, depth-1)
        turtle.forward(new_length)
        sierpinski(new_length, depth-1)
        turtle.backward(new_length)
        turtle.left(60)
        turtle.forward(new_length)
        turtle.right(60)
        sierpinski(new_length, depth-1)
        turtle.left(60)
        turtle.backward(new_length)
        turtle.right(60)

# Set up the turtle
turtle.speed(0)  # Set the speed to maximum
turtle.penup()
turtle.goto(-200, -200)
turtle.pendown()

# Draw the Sierpinski triangle
sierpinski(400, 5)

# Keep the window open until it's manually closed
turtle.done()
