# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:35:08 2023

@author: 20pt20
"""

import turtle

def koch(length, depth):
    if depth == 0:
        turtle.forward(length)
        return
    
    new_length = length / 3
    
    koch(new_length, depth-1)
    turtle.left(60)
    koch(new_length, depth-1)
    turtle.right(120)
    koch(new_length, depth-1)
    turtle.left(60)
    koch(new_length, depth-1)

# Set up the turtle
turtle.speed(0)  # Set the speed to maximum
turtle.penup()
turtle.goto(-200, 0)
turtle.pendown()

# Draw the Koch snowflake
for _ in range(3):
    koch(400, 5)
    turtle.right(120)

# Keep the window open until it's manually closed
turtle.done()
