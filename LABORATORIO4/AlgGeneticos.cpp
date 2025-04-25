
#include <GL/glut.h>
#include <vector>
#include <utility>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>

using namespace std;

// --- Datos del grafo dibujado por el usuario ---
vector<pair<int, int>> puntos;
vector<pair<int, int>> edges;
vector<double> avgDistances;


// --- Parámetros del GA ---
const int POP_SIZE    = 20;
const int GENERATIONS = 30;
const double MUT_RATE = 0.1;  // tasa de mutación por inserción
vector<vector<int>> population;
vector<vector<int>> bestTours;  // mejor tour de cada generación
vector<double> bestDistances;   // distancia del mejor tour por generación
vector<vector<double>> distMat;
bool gaDone = false;
int currentGen = 0;

// --- Funciones GA ---
double tourDistance(const vector<int>& tour) {
	double d = 0.0;
	for (int i = 0; i < (int)tour.size(); ++i) {
		int a = tour[i];
		int b = tour[(i+1) % tour.size()];
		d += distMat[a][b];
	}
	return d;
}

// PMX Crossover
vector<int> pmx(const vector<int>& p1, const vector<int>& p2) {
	int n = p1.size();
	vector<int> c(n, -1);
	static random_device rd;
	static mt19937 gen(rd());
	uniform_int_distribution<> dis(0, n-1);
	int i = dis(gen), j = dis(gen);
	if (i > j) swap(i,j);
	
	// Copiar segmento
	for (int k = i; k <= j; ++k) c[k] = p1[k];
	
	// Mapeo
	for (int k = i; k <= j; ++k) {
		int val = p2[k];
		if (find(c.begin()+i, c.begin()+j+1, val) == c.begin()+j+1) {
			int pos = k;
			while (true) {
				int mapped = p1[pos];
				auto it = find(p2.begin(), p2.end(), mapped);
				pos = distance(p2.begin(), it);
				if (c[pos] == -1) { c[pos] = val; break; }
			}
		}
	}
	// Rellenar resto
	for (int k = 0; k < n; ++k)
		if (c[k] < 0) c[k] = p2[k];
	return c;
}

// Mutación por inserción (orden)
void mutate(vector<int>& tour) {
	static random_device rd;
	static mt19937 gen(rd());
	uniform_real_distribution<> mur(0.0,1.0);
	if (mur(gen) > MUT_RATE) return;
	
	uniform_int_distribution<> dis(0, tour.size()-1);
	int i = dis(gen), j = dis(gen);
	if (i == j) return;
	int gene = tour[i];
	tour.erase(tour.begin() + i);
	tour.insert(tour.begin() + (j % tour.size()), gene);
}

// Selección por ruleta
int rouletteSelect(const vector<double>& fitness) {
	static random_device rd;
	static mt19937 gen(rd());
	double sum = accumulate(fitness.begin(), fitness.end(), 0.0);
	uniform_real_distribution<> dis(0.0, sum);
	double r = dis(gen), accum = 0.0;
	for (int i = 0; i < (int)fitness.size(); ++i) {
		accum += fitness[i];
		if (accum >= r) return i;
	}
	return fitness.size()-1;
}

// Inicializa matriz de distancias
void computeDistMatrix() {
	int n = puntos.size();
	distMat.assign(n, vector<double>(n,0.0));
	for (int i = 0; i < n; ++i)
		for (int j = i+1; j < n; ++j) {
			double dx = puntos[i].first  - puntos[j].first;
			double dy = puntos[i].second - puntos[j].second;
			double d = sqrt(dx*dx + dy*dy);
			distMat[i][j] = distMat[j][i] = d;
	}
}

void exportResults() {
	ofstream fout("resultados.txt");
	if (!fout) {
		cerr << "Error al abrir el archivo de salida." << endl;
		return;
	}
	for (int g = 0; g < GENERATIONS; ++g)
		fout << (g+1) << " " << bestDistances[g] << " " << avgDistances[g] << "\n";
	fout.close();
	cout << "Resultados exportados a resultados.txt\n";
}


void runGA() {
	int n = puntos.size();
	if (n < 2) return;
	
	computeDistMatrix();
	
	// 1) Inicializar población
	population.clear();
	vector<int> base(n);
	iota(base.begin(), base.end(), 0);
	random_device rd; mt19937 gen(rd());
	for (int i = 0; i < POP_SIZE; ++i) {
		shuffle(base.begin(), base.end(), gen);
		population.push_back(base);
	}
	
	bestTours.clear();
	bestDistances.clear();
	avgDistances.clear();  // <-- agregado
	
	
	// 2) Iterar generaciones
	for (int g = 0; g < GENERATIONS; ++g) {
		vector<double> distances(POP_SIZE), fitness(POP_SIZE);
		for (int i = 0; i < POP_SIZE; ++i) {
			distances[i] = tourDistance(population[i]);
			fitness[i] = 1.0 / distances[i];
		}
		
		double sum = 0.0;
		for (int i = 0; i < POP_SIZE; ++i) {
			distances[i] = tourDistance(population[i]);
			sum += distances[i];
			fitness[i] = 1.0 / distances[i];
		}
		avgDistances.push_back(sum / 
							   POP_SIZE);
		
		int bestIdx = min_element(distances.begin(), distances.end()) - distances.begin();
		bestTours.push_back(population[bestIdx]);
		bestDistances.push_back(distances[bestIdx]);
		
		// Imprimir distancia en terminal
		cout << "Generación " << (g+1) << ": " << distances[bestIdx] << endl;
		
		vector<vector<int>> newPop;
		newPop.push_back(population[bestIdx]); // elitismo
		while ((int)newPop.size() < POP_SIZE) {
			int i1 = rouletteSelect(fitness);
			int i2 = rouletteSelect(fitness);
			auto child = pmx(population[i1], population[i2]);
			mutate(child);
			newPop.push_back(child);
		}
		population.swap(newPop);
	}
	
	gaDone = true;
	currentGen = 0;
	stringstream ss; ss << "Mejor tour - Gen " << (currentGen+1) << "/" << GENERATIONS;
	glutSetWindowTitle(ss.str().c_str());
	
	// ...
	
	exportResults();  // <-- Agregado aquí
	
}

// --- Dibujo y eventos GLUT ---
int win_height = 480;

void reshape_cb(int w, int h) {
	if (w==0||h==0) return;
	glViewport(0,0,w,h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0,w,0,h);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	win_height = h;
}


void display_cb() {
	glClear(GL_COLOR_BUFFER_BIT);
	
	if (!gaDone) {
		glColor3f(0,0,1);
		glLineWidth(2.0);
		glBegin(GL_LINES);
		for (auto& e : edges) {
			glVertex2i(puntos[e.first].first,  puntos[e.first].second);
			glVertex2i(puntos[e.second].first, puntos[e.second].second);
		}
		glEnd();
	} else {
		glColor3f(0,1,0);
		glLineWidth(3.0);
		glBegin(GL_LINE_LOOP);
		auto& tour = bestTours[currentGen];
		for (int idx : tour) glVertex2i(puntos[idx].first, puntos[idx].second);
		glEnd();
	}
	
	glPointSize(8.0);
	glBegin(GL_POINTS);
	glColor3f(1,0,0);
	for (auto& p : puntos) glVertex2i(p.first, p.second);
	glEnd();
	
	glutSwapBuffers();
}

void mouse_cb(int button, int state, int x, int y) {
	if (state==GLUT_DOWN && button==GLUT_LEFT_BUTTON) {
		int yy = win_height - y;
		puntos.emplace_back(x, yy);
		int newIdx = puntos.size()-1;
		for (int i = 0; i < newIdx; ++i) edges.emplace_back(i, newIdx);
		gaDone = false;
		glutPostRedisplay();
	}
}

void keyboard_cb(unsigned char key, int, int) {
	if (key=='G' || key=='g') { runGA(); glutPostRedisplay(); }
	if ((key=='N' || key=='n') && gaDone) {
		currentGen = (currentGen + 1) % GENERATIONS;
		stringstream ss; ss << "Mejor tour - Gen " << (currentGen+1) << "/" << GENERATIONS;
		glutSetWindowTitle(ss.str().c_str());
		glutPostRedisplay();
	}
}

void initialize() {
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(640,480);
	glutInitWindowPosition(100,100);
	glutCreateWindow("Conexión automática de nodos");
	glutDisplayFunc(display_cb);
	glutReshapeFunc(reshape_cb);
	glutMouseFunc(mouse_cb);
	glutKeyboardFunc(keyboard_cb);
	glClearColor(1,1,1,1);
}



int main(int argc, char** argv) {
	glutInit(&argc, argv);
	initialize();
	glutMainLoop();
	return 0;
}
