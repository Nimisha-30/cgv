#include <windows.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

int count=0;
 struct point
{
    GLint x;
    GLint y;
};
point p[50];

void drawLine(point a,point b)
{
    glBegin(GL_LINES);
    glColor3d(0.5,0.5,0.5);
    glVertex2d(a.x,a.y);
    glVertex2d(b.x,b.y);
    glEnd();
    glFlush();
}
void curve(float t)
{
    point temp;
    temp.x=(1-t)*(1-t)*(1-t)*p[0].x+3*t*(1-t)*(1-t)*p[1].x+3*t*t*(1-t)*p[2].x+t*t*t*p[3].x;
    temp.y=(1-t)*(1-t)*(1-t)*p[0].y+3*t*(1-t)*(1-t)*p[1].y+3*t*t*(1-t)*p[2].y+t*t*t*p[3].y;
    glBegin(GL_POINTS);
    glColor3f(1,0,0.5);
    glVertex2d(temp.x,temp.y);
    glEnd();
    glFlush();
}
void mouse(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        if(count==0)
        {
            p[count].x=x;
            p[count].y=480-y;
            count=count+1;
        }
        else if(count<4)
        {
            p[count].x=x;
            p[count].y=480-y;
            count=count+1;
            drawLine(p[count-2],p[count-1]);
        }
        else
        {
            count=0;
            float t=0.001;
            for(int i=0;t*i<1;i++)
            {
                curve(t*i);
            }
        }
	}
	else if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
        {
            ;
        }
}
void display()
{

}
void init()
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,640,0,480);
    glClearColor(0,0,0,0);
}

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(100,100);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);

    glutCreateWindow("GLUT Shapes");
    init();

    glutMouseFunc(mouse);
    glutDisplayFunc(display);
    glutMainLoop();
    return EXIT_SUCCESS;
}


