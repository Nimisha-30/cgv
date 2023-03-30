import turtle

def levy_curve(t, iterations, length, shortening_factor, angle):
    if iterations == 0:
        t.forward(length)
    else:
        iterations = iterations - 1
        length = length / shortening_factor
        t.left(angle)
        levy_curve(t, iterations, length, shortening_factor, angle)
        t.right(angle * 2)
        levy_curve(t, iterations, length, shortening_factor, angle)
        t.left(angle)

if __name__ =="__main__":
    t = turtle.Turtle()
    t.hideturtle()
    levy_curve(t, 7, 100, pow(2,0.5), 45)