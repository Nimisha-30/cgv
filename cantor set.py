# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:29:54 2023

@author: 20pt20
"""

import turtle

def cantor(x, y, length, depth):
    if depth == 0:
        return
    
    # Draw the current segment
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()
    turtle.forward(length)
    
    # Calculate the length of the next segments
    new_length = length / 3
    
    # Draw the two smaller segments
    cantor(x, y-20, new_length, depth-1)
    cantor(x+2*new_length, y-20, new_length, depth-1)

# Set up the turtle
turtle.speed(0)  # Set the speed to maximum
turtle.penup()
turtle.goto(-200, 0)
turtle.pendown()

# Draw the Cantor set
cantor(-200, 0, 400, 5)

# Keep the window open until it's manually closed
turtle.done()
