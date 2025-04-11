#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>

using namespace std;

vector<pair<int,int>> puntos;
vector<pair<int,int>> unir;
vector<pair<int,int>> seleccionados;
pair<int,int> act;
int cv = 0;

void mouseCallBack(int, int, int, int);
bool detectarPunto(int, int);

int win_height = 480;

void reshape_cb(int w, int h) {
	if (w == 0 || h == 0) return;
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	win_height = h;
}

void display_cb() {
	glClear(GL_COLOR_BUFFER_BIT);
	
	glColor3f(0, 0, 1);
	glLineWidth(3.0);
	glBegin(GL_LINES);
	for (size_t i = 0; i + 1 < unir.size(); i += 2) 
	{
		glVertex2i(unir[i].first, unir[i].second);
		glVertex2i(unir[i + 1].first, unir[i + 1].second);
	}
	glEnd();
	
	glColor3f(1, 0, 0);
	glPointSize(30.0);
	glBegin(GL_POINTS);
	for (auto &p : puntos)
		glVertex2i(p.first, p.second);
	glEnd();
	



	
	glutSwapBuffers();
}

void initialize() {
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Ventana OpenGL");
	glutDisplayFunc(display_cb);
	glutReshapeFunc(reshape_cb);
	glutMouseFunc(mouseCallBack);
	glClearColor(1.f, 1.f, 1.f, 1.f);
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	initialize();
	glutMainLoop();
	return 0;
}

bool detectarPunto(int x, int y) {
	for (auto &a : puntos) {
		if (x < a.first + 15 && x > a.first - 15)
			if (y < a.second + 15 && y > a.second - 15)
				return true;
	}
	return false;
}

void mouseCallBack(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		act.first = x;
		act.second = win_height - y;
		
		if (cv < 6) {
			puntos.push_back(act);
			cv++;
		} else {
			if (detectarPunto(x, win_height - y)) {
				seleccionados.push_back(act);
				if (seleccionados.size() == 2) {
					unir.push_back(seleccionados[0]);
					unir.push_back(seleccionados[1]);
					seleccionados.clear();
				}
			}
		}
		
		cout << x << " , " << y << "\n";
		glutPostRedisplay();
	}
}
