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
    }
    void crearHijos(char C) {
        vector<vector<char>> tab2 = tablero;
        for (int i = 0; i < tab2.size(); i++) {
            for (int j = 0; j < tab2.size(); j++) {
                if (tab2[i][j] == ' ') {
                    tab2[i][j] = C;
                    Nodo aux(tab2);
                    hijos.push_back(aux);
                    tab2[i][j] = ' ';
                }
            }
        }
    }
    bool ganador() {
        return quienGano() != ' ';
    }
    char hayGanador() {
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
                if (igual) return primero;
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
                if (igual) return primero;
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
            if (igual) return diag;
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
            if (igual) return diag;
        }

        return ' '; 
    }
    char quienGano() {
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
                if (igual) return primero;
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
                if (igual) return primero;
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
            if (igual) return diag;
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
            if (igual) return diag;
        }

        return ' ';
    }
    int contarOpciones(char jugador, char ia) {
        int n = tablero.size();
        int opcionesIA = 0, opcionesJugador = 0;

        // revisa filas
        for (int i = 0; i < n; i++) {
            bool posibleIA = true, posibleJ = true;
            for (int j = 0; j < n; j++) {
                if (tablero[i][j] == jugador) posibleIA = false;
                if (tablero[i][j] == ia) posibleJ = false;
            }
            if (posibleIA) opcionesIA++;
            if (posibleJ) opcionesJugador++;
        }

        // revisa columnass
        for (int j = 0; j < n; j++) {
            bool posibleIA = true, posibleJ = true;
            for (int i = 0; i < n; i++) {
                if (tablero[i][j] == jugador) posibleIA = false;
                if (tablero[i][j] == ia) posibleJ = false;
            }
            if (posibleIA) opcionesIA++;
            if (posibleJ) opcionesJugador++;
        }

        // diagonal principal
        bool posibleIA = true, posibleJ = true;
        for (int i = 0; i < n; i++) {
            if (tablero[i][i] == jugador) posibleIA = false;
            if (tablero[i][i] == ia) posibleJ = false;
        }
        if (posibleIA) opcionesIA++;
        if (posibleJ) opcionesJugador++;

        // dagonal secundaria
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
};

class Arbol {
    int deepth;

public:
    Nodo* raiz;
    Arbol(int d, vector<vector<char>> tab) : deepth(d) {
        raiz = new Nodo(tab);
    }
    void minimax(char jugador, char ia, int profundidad, bool turnoIa, int alpha, int beta) {
        char ganadorActual = raiz->hayGanador();
        if (profundidad == 0 || ganadorActual != ' ') {
            if (ganadorActual == ia)
                raiz->puntuacion = 1000;
            else if (ganadorActual == jugador)
                raiz->puntuacion = -1000;
            else
                raiz->puntuacion = raiz->contarOpciones(jugador, ia);
            return;
        }

        char simbolo = turnoIa ? ia : jugador;
        raiz->crearHijos(simbolo);

        if (raiz->hijos.empty()) {
            raiz->puntuacion = raiz->contarOpciones(jugador, ia);
            return;
        }

        if (turnoIa) {
            int maxPunt = -10000;
            for (auto& hijo : raiz->hijos) {
                Arbol subArbol(profundidad - 1, hijo.tablero);
                subArbol.minimax(jugador, ia, profundidad - 1, false, alpha, beta);
                hijo.puntuacion = subArbol.raiz->puntuacion;
                maxPunt = max(maxPunt, hijo.puntuacion);
                alpha = max(alpha, hijo.puntuacion);
                if (beta <= alpha) break;  
            }
            raiz->puntuacion = maxPunt;
        }
        else {
            int minPunt = 10000;
            for (auto& hijo : raiz->hijos) {
                Arbol subArbol(profundidad - 1, hijo.tablero);
                subArbol.minimax(jugador, ia, profundidad - 1, true, alpha, beta);
                hijo.puntuacion = subArbol.raiz->puntuacion;
                minPunt = min(minPunt, hijo.puntuacion);
                beta = min(beta, hijo.puntuacion);
                if (beta <= alpha) break;  
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
    vector<vector<char>> tablero;

public:
    Juego() {
        while (1) {
            cout << "Escoge tu ficha X (1) o O(2) (X siempre empieza)\n>>> ";
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

        cout << "Escoge el tamano del tablero.\n>>> ";
        cin >> n;
        cout << "Escoge la profundidad de la busqueda.\n>>> ";
        cin >> deep;

        empezar(n, deep);
    }

    void prin() {
        for (int i = 0; i < tablero.size(); i++) {
            cout << "|";
            for (int j = 0; j < tablero[i].size(); j++) {
                cout << tablero[i][j] << "|";
            }
            cout << "\n";
        }
        cout << endl;
    }
    void empezar(int n, int d) {
        tablero.resize(n, vector<char>(n, ' '));
        partida();
    }
    void marcar(int x, int y) {
        if (tablero[x][y] == ' ') {
            tablero[x][y] = jugador;
        }
        else {
            cout << "Esa casilla ya está ocupada.\n";
        }
    }
    void juegaIA() {
        A = new Arbol(deep, tablero);
        A->minimax(jugador, ia, deep, true, -10000, 10000);


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
    }
    void partida() {
        Nodo evaluador(tablero);
        int x, y;

        while (!tableroLleno() && evaluador.quienGano() == ' ') {
            cout << "coords (fila columna):\n>>> ";
            cin >> x >> y;
            marcar(x, y);
            prin();

            evaluador = Nodo(tablero);
            if (evaluador.quienGano() != ' ') break;

            juegaIA();
            prin();

            evaluador = Nodo(tablero);
        }

        char resultado = evaluador.quienGano();
        if (resultado == jugador)
            cout << "¡Has ganado!\n";
        else if (resultado == ia)
            cout << "¡La IA ha ganado!\n";
        else
            cout << "¡Empate!\n";
    }
    bool tableroLleno() {
        for (auto& fila : tablero) {
            for (char c : fila) {
                if (c == ' ') return false;
            }
        }
        return true;
    }
};

int main() {
    Juego game;
    return 0;
}
