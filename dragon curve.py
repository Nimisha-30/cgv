# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:35:14 2023

@author: 20pt20
"""

import turtle

def dragon(length, depth, direction):
    if depth == 0:
        turtle.forward(length)
        return
    
    new_length = length / 2 ** 0.5
    
    turtle.right(direction * 45)
    dragon(new_length, depth-1, 1)
    turtle.left(direction * 90)
    dragon(new_length, depth-1, -1)
    turtle.right(direction * 45)

# Set up the turtle
turtle.speed(0)  # Set the speed to maximum
turtle.penup()
turtle.goto(-200, 0)
turtle.pendown()

# Draw the Dragon Curve
dragon(400, 10, 1)

# Keep the window open until it's manually closed
turtle.done()
