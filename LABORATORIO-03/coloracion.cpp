#include <GL/glut.h>
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>

using namespace std;

vector<pair<int,int>> puntos;
vector<pair<int,int>> edges;
vector<int> seleccionadosIndices;
pair<int,int> act;
int cv = 0;
int win_height = 480;
vector<int> vertexColors;

struct Color {
	float r, g, b;
};
vector<Color> palette = {
	{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
	{1, 0, 1}, {1, 0.5f, 0}, {0.5f, 0, 0.5f}
};

int findVertexIndex(int x, int y) {
	for (int i = 0; i < (int)puntos.size(); i++) {
		int dx = x - puntos[i].first;
		int dy = y - puntos[i].second;
		if (abs(dx) <= 15 && abs(dy) <= 15)
			return i;
	}
	return -1;
}

vector<vector<int>> buildGraph() {
	int n = puntos.size();
	vector<vector<int>> adj(n);
	for (auto &e : edges) {
		int idx1 = e.first;
		int idx2 = e.second;
		if (find(adj[idx1].begin(), adj[idx1].end(), idx2) == adj[idx1].end())
			adj[idx1].push_back(idx2);
		if (find(adj[idx2].begin(), adj[idx2].end(), idx1) == adj[idx2].end())
			adj[idx2].push_back(idx1);
	}
	return adj;
}

void greedyColoring(const vector<int>& order, const vector<vector<int>>& adj) {
	int n = puntos.size();
	vertexColors.assign(n, -1);
	for (int v : order) {
		vector<bool> used(n, false);
		for (int neighbor : adj[v]) {
			if (vertexColors[neighbor] != -1)
				used[ vertexColors[neighbor] ] = true;
		}
		int color = 0;
		while (color < n && used[color]) color++;
		vertexColors[v] = color;
	}
}

void colorGraphMaxPriority() {
	vector<vector<int>> adj = buildGraph();
	int n = puntos.size();
	vector<int> order(n);
	for (int i = 0; i < n; i++) order[i] = i;
	sort(order.begin(), order.end(), [&adj](int a, int b) {
		return adj[a].size() > adj[b].size();
	});
	greedyColoring(order, adj);
	cout << "Max priority:\n";
	for (int i = 0; i < n; i++)
		cout << "V" << i << " -> C" << vertexColors[i] << "\n";
	glutPostRedisplay();
}

void colorGraphMinPriority() {
	vector<vector<int>> adj = buildGraph();
	int n = puntos.size();
	vector<int> order(n);
	for (int i = 0; i < n; i++) order[i] = i;
	sort(order.begin(), order.end(), [&adj](int a, int b) {
		return adj[a].size() < adj[b].size();
	});
	greedyColoring(order, adj);
	cout << "Min priority:\n";
	for (int i = 0; i < n; i++)
		cout << "V" << i << " -> C" << vertexColors[i] << "\n";
	glutPostRedisplay();
}

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
	for (auto &e : edges) {
		int idx1 = e.first;
		int idx2 = e.second;
		glVertex2i(puntos[idx1].first, puntos[idx1].second);
		glVertex2i(puntos[idx2].first, puntos[idx2].second);
	}
	glEnd();
	
	glPointSize(30.0);
	glBegin(GL_POINTS);
	for (size_t i = 0; i < puntos.size(); i++) {
		int c = (i < vertexColors.size() && vertexColors[i] != -1) ? vertexColors[i] : 0;
		Color col = palette[c % palette.size()];
		glColor3f(col.r, col.g, col.b);
		glVertex2i(puntos[i].first, puntos[i].second);
	}
	glEnd();
	
	glutSwapBuffers();
}

void keyboard_cb(unsigned char key, int x, int y) {
	if (key == 'm') colorGraphMaxPriority();
	else if (key == 'n') colorGraphMinPriority();
}

void mouseCallBack(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		act.first = x;
		act.second = win_height - y;
		if (cv < 6) {
			puntos.push_back(act);
			cv++;
		} else {
			int idx = findVertexIndex(x, win_height - y);
			if (idx != -1) {
				seleccionadosIndices.push_back(idx);
				if (seleccionadosIndices.size() == 2) {
					edges.push_back({seleccionadosIndices[0], seleccionadosIndices[1]});
					seleccionadosIndices.clear();
				}
			}
		}
		glutPostRedisplay();
	}
}

void initialize() {
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Ventana OpenGL - Coloracion de Grafos");
	glutDisplayFunc(display_cb);
	glutReshapeFunc(reshape_cb);
	glutMouseFunc(mouseCallBack);
	glutKeyboardFunc(keyboard_cb);
	glClearColor(1.f, 1.f, 1.f, 1.f);
}

int main(int argc, char **argv) {
	glutInit(&argc, argv);
	initialize();
	glutMainLoop();
	return 0;
}

