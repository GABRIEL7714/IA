#include <iostream>
#include <vector>
using namespace std;

class Nodo {

public:
	vector<Nodo> hijos;
	vector<vector<char>> tablero;
	int puntuacion;


	Nodo(vector<vector<char>> tab) {
		puntuacion = 0;
		tablero = tab;
	};

	const vector<vector<char>>& getTablero() const {
		return tablero;
	}


	void crearHijos(char C) {
		vector<vector<char>> tab2 = tablero;
		for (int i{ 0 }; i < tab2.size(); i++) {
			for (int j{ 0 }; j < tab2.size(); j++) {
				if (tab2[i][j] == ' ') {
					tab2[i][j] = C;
					Nodo aux(tab2);
					hijos.push_back(aux);
					tab2[i][j] = ' ';
				}
			}
		}
	}

	int contarOpciones(char jugador, char ia) {
		int n = tablero.size();
		int opcionesIA = 0, opcionesJugador = 0;

		// Revisa filas
		for (int i = 0; i < n; i++) {
			bool posibleIA = true, posibleJ = true;
			for (int j = 0; j < n; j++) {
				if (tablero[i][j] == jugador) posibleIA = false;
				if (tablero[i][j] == ia) posibleJ = false;
			}
			if (posibleIA) opcionesIA++;
			if (posibleJ) opcionesJugador++;
		}

		// Revisa columnas
		for (int j = 0; j < n; j++) {
			bool posibleIA = true, posibleJ = true;
			for (int i = 0; i < n; i++) {
				if (tablero[i][j] == jugador) posibleIA = false;
				if (tablero[i][j] == ia) posibleJ = false;
			}
			if (posibleIA) opcionesIA++;
			if (posibleJ) opcionesJugador++;
		}

		// Revisa diagonal principal
		bool posibleIA = true, posibleJ = true;
		for (int i = 0; i < n; i++) {
			if (tablero[i][i] == jugador) posibleIA = false;
			if (tablero[i][i] == ia) posibleJ = false;
		}
		if (posibleIA) opcionesIA++;
		if (posibleJ) opcionesJugador++;

		// Revisa diagonal secundaria
		posibleIA = true, posibleJ = true;
		for (int i = 0; i < n; i++) {
			if (tablero[i][n - 1 - i] == jugador) posibleIA = false;
			if (tablero[i][n - 1 - i] == ia) posibleJ = false;
		}
		if (posibleIA) opcionesIA++;
		if (posibleJ) opcionesJugador++;

		puntuacion = opcionesIA - opcionesJugador;
		return puntuacion;
	}


	bool ganador() {
		int n = tablero.size();


		for (int i = 0; i < n; ++i) {
			char primero = tablero[i][0];
			if (primero != ' ') {
				bool igual = true;
				for (int j = 1; j < n; ++j) {
					if (tablero[i][j] != primero) {
						igual = false;
						break;
					}
				}
				if (igual) return true;
			}
		}

		for (int j = 0; j < n; ++j) {
			char primero = tablero[0][j];
			if (primero != ' ') {
				bool igual = true;
				for (int i = 1; i < n; ++i) {
					if (tablero[i][j] != primero) {
						igual = false;
						break;
					}
				}
				if (igual) return true;
			}
		}

		char diag = tablero[0][0];
		if (diag != ' ') {
			bool igual = true;
			for (int i = 1; i < n; ++i) {
				if (tablero[i][i] != diag) {
					igual = false;
					break;
				}
			}
			if (igual) return true;
		}

		diag = tablero[0][n - 1];
		if (diag != ' ') {
			bool igual = true;
			for (int i = 1; i < n; ++i) {
				if (tablero[i][n - 1 - i] != diag) {
					igual = false;
					break;
				}
			}
			if (igual) return true;
		}

		return false;
	}
};

class Arbol {
	int deepth;

public:

	Nodo* raiz;

	Arbol(int d, vector<vector<char>>tab) : deepth(d) {

		raiz = new Nodo(tab);
	};


	void minimax(char jugador, char ia, int profundidad, bool turnoIa) {
		if (profundidad == 0 || raiz->ganador()) {
			raiz->puntuacion = raiz->contarOpciones(jugador,ia);
			return;
		}

		char simbolo = turnoIa ? ia : jugador;
		raiz->crearHijos(simbolo);

		if (raiz->hijos.empty()) {
			raiz->puntuacion = raiz->contarOpciones(jugador,ia);
			return;
		}

		for (auto& hijo : raiz->hijos) {
			Arbol subArbol(profundidad - 1, hijo.tablero);
			subArbol.minimax(jugador, ia, profundidad - 1, !turnoIa);
			hijo.puntuacion = subArbol.raiz->puntuacion;
		}

		if (turnoIa) {
			int maxPunt = -1e9;
			//int iMax;
			for (int i = 0; i < raiz->hijos.size(); i++) {
				if (raiz->hijos[i].puntuacion > maxPunt) {
					maxPunt = raiz->hijos[i].puntuacion;
					//iMax = i;
				}
			}
			raiz->puntuacion = maxPunt;
			/*if (profundidad == deepth) {
				raiz->tablero = raiz->hijos[iMax].tablero;
			}*/
		}
		else {
			int minPunt = 1e9;
			for (auto& hijo : raiz->hijos) {
				minPunt = min(minPunt, hijo.puntuacion);
			}
			raiz->puntuacion = minPunt;
		}
	}



};

class Juego {
	char jugador;
	char ia;
	int deep;
	int n;
	Arbol* A;

	int turno;
	vector<vector<char>> tablero;

public:

	Juego() {
		while (1) {
			cout << "Escoge tu ficha X (1) o O(2) (X siempre empieza)\n>>>";
			int opt;
			cin >> opt;
			if (opt == 1) {
				jugador = 'X';
				ia = 'O';
				break;
			}
			else if (opt == 2) {
				jugador = 'O';
				ia = 'X';
				break;
			}
			else {
				cout << "Opcion no valida.\n";
			}
		}

		cout << "Escoge el tamano del tablero.\n>>>";
		cin >> n;
		cout << "Escoge la profundidad de la busqueda.\n>>>";
		cin >> deep;
		empezar(n, deep);
		turno = 0;
	}

	void prin() {
		for (int i{ 0 }; i < tablero.size(); i++) {
			cout << "|";
			for (int j{ 0 }; j < tablero[i].size(); j++) {
				cout << tablero[i][j] << "|";
			}
			cout << "\n";
		}
		cout << endl;

	}

	void empezar(int n, int d) {

		tablero.resize(n);
		for (int i{ 0 }; i < tablero.size(); i++) {
			tablero[i].resize(n);
		}

		for (int i{ 0 }; i < tablero.size(); i++) {
			for (int j{ 0 }; j < tablero[i].size(); j++) {
				tablero[i][j] = ' ';
			}
		}
		//prin();
		partida();
	}

	void marcar(int x, int y) {

		if (turno % 2 == 0) {
			tablero[x][y] = jugador;
		}
		else {
			tablero[x][y] = ia;
		}

	}

	void juegaIA() {
		A->minimax(jugador, ia, deep, 1);

		int mejor = -10000;
		int indice = -1;
		for (int i = 0; i < A->raiz->hijos.size(); i++) {
			if (A->raiz->hijos[i].puntuacion > mejor) {
				mejor = A->raiz->hijos[i].puntuacion;
				indice = i;
			}
		}

		if (indice != -1) {
			tablero = A->raiz->hijos[indice].tablero;
		}
		A->raiz->hijos.clear();

	}

	void partida() {
		A = new Arbol(deep, tablero);
		//cout << "coords\n>>>";
		int x, y;

		while (!tableroLleno()) {
			cout << "coords\n>>>";
			cin >> x >> y;
			marcar(x, y);
			prin();
			A->raiz->tablero= tablero;
			juegaIA();
			prin();
		}
	}
	//bool detectarGanador(){return 0;}
	bool tableroLleno() {
		bool lleno = true;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (tablero[i][j] == ' ') {
					lleno = false;
				}
			}
		}

		return lleno;
	}

};


int main() {

	Juego game;
	return 0;

}
