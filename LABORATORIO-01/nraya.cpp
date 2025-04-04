#include <GL/glut.h>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
using namespace std;

int n, enLinea, windowSize = 600;
char turno;
bool gameOver = false;
vector<vector<char>> tablero;

bool checkVictory(vector<vector<char>>& board, char player);
bool tableroLleno(vector<vector<char>>&);
class Nodo {
public:
	vector<vector<char>> tablero;
	char jugador;
	int valor;
	vector<Nodo*> hijos;
	
	Nodo(vector<vector<char>> tab, char jug) : tablero(tab), jugador(jug), valor(0) {}
};

class Arbol {
public:
	Nodo* raiz;
	
	Arbol(vector<vector<char>> estadoInicial, char jugadorInicial) {
		raiz = new Nodo(estadoInicial, jugadorInicial);
	}
	
	int minimax(Nodo* nodo, bool esMaximizador) {
		if (checkVictory(nodo->tablero, 'X')) return -10;
		if (checkVictory(nodo->tablero, 'O')) return 10;
		if (tableroLleno(nodo->tablero)) return 0;
		
		if (esMaximizador) {
			int mejorValor = -1000;
			generarHijos(nodo, 'O');
			for (Nodo* hijo : nodo->hijos) {
				mejorValor = max(mejorValor, minimax(hijo, false));
			}
			nodo->valor = mejorValor;
			return mejorValor;
		} else {
			int mejorValor = 1000;
			generarHijos(nodo, 'X');
			for (Nodo* hijo : nodo->hijos) {
				mejorValor = min(mejorValor, minimax(hijo, true));
			}
			nodo->valor = mejorValor;
			return mejorValor;
		}
	}
	
	void generarHijos(Nodo* nodo, char proximoJugador) {
		int tam = nodo->tablero.size();
		for (int i = 0; i < tam; ++i) {
			for (int j = 0; j < tam; ++j) {
				if (nodo->tablero[i][j] == ' ') {
					vector<vector<char>> nuevoTablero = nodo->tablero;
					nuevoTablero[i][j] = proximoJugador;
					Nodo* hijo = new Nodo(nuevoTablero, proximoJugador);
					nodo->hijos.push_back(hijo);
				}
			}
		}
	}
	
	pair<int, int> obtenerMejorMovimiento() {
		int mejorValor = -1000;
		pair<int, int> mejorMovimiento = {-1, -1};
		generarHijos(raiz, 'O');
		for (Nodo* hijo : raiz->hijos) {
			int val = minimax(hijo, false);
			if (val > mejorValor) {
				mejorValor = val;
				mejorMovimiento = encontrarDiferencia(raiz->tablero, hijo->tablero);
			}
		}
		return mejorMovimiento;
	}
	
	pair<int, int> encontrarDiferencia(vector<vector<char>> a, vector<vector<char>> b) {
		int tam = a.size();
		for (int i = 0; i < tam; ++i)
			for (int j = 0; j < tam; ++j)
				if (a[i][j] != b[i][j])
					return {i, j};
		return {-1, -1};
	}
};

void dibujarTablero();
void turnoComputadora();

void initTablero() {
	tablero = vector<vector<char>>(n, vector<char>(n, ' '));
}

void display() {
	glClear(GL_COLOR_BUFFER_BIT);
	dibujarTablero();
	glFlush();
}

void dibujarTablero() {
	float cellSize = 2.0f / n;
	glColor3f(0, 0, 0);
	glLineWidth(2.0);
	
	for (int i = 1; i < n; ++i) {
		float pos = -1.0f + i * cellSize;
		glBegin(GL_LINES);
		glVertex2f(pos, -1.0f);
		glVertex2f(pos, 1.0f);
		glEnd();
		
		glBegin(GL_LINES);
		glVertex2f(-1.0f, pos);
		glVertex2f(1.0f, pos);
		glEnd();
	}
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			float x = -1.0f + j * cellSize + cellSize / 2;
			float y = 1.0f - i * cellSize - cellSize / 2;
			if (tablero[i][j] == 'X') {
				glColor3f(1, 0, 0);
				glBegin(GL_LINES);
				glVertex2f(x - cellSize / 4, y - cellSize / 4);
				glVertex2f(x + cellSize / 4, y + cellSize / 4);
				glVertex2f(x - cellSize / 4, y + cellSize / 4);
				glVertex2f(x + cellSize / 4, y - cellSize / 4);
				glEnd();
			} else if (tablero[i][j] == 'O') {
				glColor3f(0, 0, 1);
				glBegin(GL_LINE_LOOP);
				for (int k = 0; k < 100; ++k) {
					float theta = 2.0f * 3.1415926f * float(k) / float(100);
					float dx = cellSize / 4 * cosf(theta);
					float dy = cellSize / 4 * sinf(theta);
					glVertex2f(x + dx, y + dy);
				}
				glEnd();
			}
		}
	}
}

bool checkVictory(vector<vector<char>>& board, char player) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j <= n - enLinea; ++j) {
			bool fila = true, col = true;
			for (int k = 0; k < enLinea; ++k) {
				if (board[i][j + k] != player) fila = false;
				if (board[j + k][i] != player) col = false;
			}
			if (fila || col) return true;
		}
	}
	for (int i = 0; i <= n - enLinea; ++i) {
		for (int j = 0; j <= n - enLinea; ++j) {
			bool diag1 = true, diag2 = true;
			for (int k = 0; k < enLinea; ++k) {
				if (board[i + k][j + k] != player) diag1 = false;
				if (board[i + k][j + enLinea - 1 - k] != player) diag2 = false;
			}
			if (diag1 || diag2) return true;
		}
	}
	return false;
}

bool tableroLleno(vector<vector<char>>& board) {
	for (auto& fila : board)
		for (auto& celda : fila)
			if (celda == ' ') return false;
	return true;
}

void turnoComputadora() {
	if (gameOver) return;
	Arbol arbol(tablero, 'O');
	pair<int, int> mov = arbol.obtenerMejorMovimiento();
	int i = mov.first;
	int j = mov.second;
	
	if (i != -1) {
		tablero[i][j] = 'O';
		if (checkVictory(tablero, 'O')) {
			cout << "\n¡La computadora gana!\n";
			gameOver = true;
		} else if (tableroLleno(tablero)) {
			cout << "\n¡Empate!\n";
			gameOver = true;
		} else {
			turno = 'X';
		}
	}
}

void computadoraCallback(int value) {
	turnoComputadora();
	glutPostRedisplay();
}

void mouseClick(int button, int state, int x, int y) {
	if (gameOver || turno != 'X') return;
	
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cellSize = windowSize / float(n);
		int j = int(x / cellSize);
		int i = int(y / cellSize);
		i = n - 1 - i;
		
		if (i >= 0 && i < n && j >= 0 && j < n && tablero[i][j] == ' ') {
			tablero[i][j] = 'X';
			if (checkVictory(tablero, 'X')) {
				cout << "\n¡Ganaste!\n";
				gameOver = true;
			} else if (tableroLleno(tablero)) {
				cout << "\n¡Empate!\n";
				gameOver = true;
			} else {
				turno = 'O';
				glutTimerFunc(300, computadoraCallback, 0);
			}
		}
	}
	glutPostRedisplay();
}

void reshape(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv) {
	cout << "Dimensión del tablero (n x n): ";
	cin >> n;
	cout << "¿Cuántas en línea para ganar?: ";
	cin >> enLinea;
	cout << "¿Quién empieza? (X/jugador o O/IA): ";
	cin >> turno;
	
	initTablero();
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(windowSize, windowSize);
	glutCreateWindow("N en línea con Minimax y arbol de decisión");
	glutDisplayFunc(display);
	glutMouseFunc(mouseClick);
	glutReshapeFunc(reshape);
	
	glClearColor(1, 1, 1, 1);
	
	if (turno == 'O') {
		turnoComputadora();
	}
	
	glutMainLoop();
	return 0;
}
