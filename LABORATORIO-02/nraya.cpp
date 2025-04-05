
#include <iostream>
#include <vector>
using namespace std;

class Nodo {
	vector<vector<char>> tablero;
	vector<Nodo*> hijos;

public:
	Nodo(vector<vector<char>> tab) {
	
		tablero = tab;
	};



	int contarOpciones();

	bool ganador();
};

class Arbol {

	int deepth;
	int n;

public:
	
	Nodo* raiz;


	Arbol(int d, vector<vector<char>> tab) : deepth(d) {
	
		raiz = new Nodo(tab);
	};


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
		int n;
		cin >> n;
		cout << "Escoge la profundidad de la busqueda.\n>>>";
		int d;
		cin >> d;
		empezar(n, d); 
		deep = d;
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
		prin();
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

		A = new Arbol(deep,tablero);

	}

	void partida() {
		cout << "coords\n>>>";
		int x, y;
		cin >> x >> y;
		marcar(x,y);
		prin();
		

	}

};


int main() {

	Juego game;



}
