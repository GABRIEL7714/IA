#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>


using namespace std;

vector<pair<int,int>> puntos;
vector<pair<int,int>> unir;
pair<int,int> act;


void mouseCallBack(int,int,int,int);

void reshape_cb (int w, int h) {
	if (w==0||h==0) return;
	glViewport(0,0,w,h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D(0,w,0,h);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
}

void display_cb() {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,0,0); 
	glPointSize(30.0);
	glBegin(GL_POINTS);
	for(auto &p : puntos)glVertex2i(p.first, p.second);
	glEnd();

	glutSwapBuffers();
}

void initialize() {
	glutInitDisplayMode (GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowSize (640,480);
	glutInitWindowPosition (100,100);
	glutCreateWindow ("Ventana OpenGL");
	glutDisplayFunc (display_cb);
	glutReshapeFunc (reshape_cb);
	glClearColor(1.f,1.f,1.f,1.f);
	
	glutMouseFunc(mouseCallBack);
}

int main (int argc, char **argv) {
	glutInit (&argc, argv);
	initialize();
	glutMainLoop();
	return 0;
}

void mouseCallBack(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		act.first=x;
		act.second =abs(y-480);
		puntos.push_back(act);
		cout << x << " , " << y  << "\n";
	}
	
}

if

