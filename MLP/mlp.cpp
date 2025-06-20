#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace Eigen;

// Leer CSV completo en memoria
auto readCSV(const string& filename) {
    vector<vector<float>> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return data;
    }
    string line;
    while (getline(file, line)) {
        vector<float> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            try { row.push_back(stof(cell)); }
            catch (...) { return vector<vector<float>>(); }
        }
        data.push_back(move(row));
    }
    return data;
}

MatrixXd sigmoid(const MatrixXd& x) { return 1.0 / (1.0 + (-x.array()).exp()); }
MatrixXd sigmoidDerivative(const MatrixXd& x) { return x.array() * (1.0 - x.array()); }

MatrixXd softmax(const MatrixXd& z) {
    MatrixXd e = z.array().exp();
    return e.array().rowwise() / e.colwise().sum().array();
}

class PerceptronMulticapa {
public:
    PerceptronMulticapa(int inSize, int hidSize, int outSize);
    void train(const MatrixXd& X, const MatrixXd& y, int epochs, double lr);
    VectorXd predict(const VectorXd& x);
private:
    MatrixXd W1, W2;
    VectorXd b1, b2;
    vector<double> erroresPorEpoca;
    MatrixXd forward(const MatrixXd& X);
    double calcError(const MatrixXd& out, const MatrixXd& y);
    void backward(const MatrixXd& X, const MatrixXd& y, const MatrixXd& out, double lr);
    double calcAcc(const MatrixXd& X, const MatrixXd& y);
    void exportarErrores();
};

PerceptronMulticapa::PerceptronMulticapa(int inSize, int hidSize, int outSize) {
    srand((unsigned)time(0));
    W1 = MatrixXd::NullaryExpr(hidSize, inSize, [&]() { return ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / inSize); });
    W2 = MatrixXd::NullaryExpr(outSize, hidSize, [&]() { return ((double)rand() / RAND_MAX - 0.5) * sqrt(2.0 / hidSize); });
    b1 = VectorXd::Zero(hidSize);
    b2 = VectorXd::Zero(outSize);
}

void PerceptronMulticapa::train(const MatrixXd& X, const MatrixXd& y, int epochs, double lr) {
    for (int e = 0; e < epochs; ++e) {
        auto out = forward(X);
        backward(X, y, out, lr);
        double error = calcError(out, y);
        erroresPorEpoca.push_back(error);
        cout << "Epoch " << e + 1 << " Acc: " << calcAcc(X, y) * 100 << "% - Error: " << error << endl;
    }
    exportarErrores();
}

MatrixXd PerceptronMulticapa::forward(const MatrixXd& X) {
    MatrixXd Z1 = (W1 * X.transpose()).colwise() + b1;
    MatrixXd A1 = sigmoid(Z1);
    MatrixXd Z2 = (W2 * A1).colwise() + b2;
    return softmax(Z2);
}

double PerceptronMulticapa::calcError(const MatrixXd& out, const MatrixXd& y) {
    MatrixXd diff = out.transpose() - y;
    return diff.array().square().sum();
}

void PerceptronMulticapa::backward(const MatrixXd& X, const MatrixXd& y, const MatrixXd& out, double lr) {
    MatrixXd err = out - y.transpose();
    MatrixXd A1 = sigmoid((W1 * X.transpose()).colwise() + b1);
    MatrixXd d2 = err; // softmax + MSE
    MatrixXd d1 = (W2.transpose() * d2).cwiseProduct(sigmoidDerivative(A1));
    W2 -= lr * d2 * A1.transpose();
    b2 -= lr * d2.rowwise().sum();
    W1 -= lr * d1 * X;
    b1 -= lr * d1.rowwise().sum();
}

VectorXd PerceptronMulticapa::predict(const VectorXd& x) {
    auto Z1 = (W1 * x).colwise() + b1;
    auto A1 = sigmoid(Z1);
    auto Z2 = (W2 * A1).colwise() + b2;
    return softmax(Z2).col(0);
}

double PerceptronMulticapa::calcAcc(const MatrixXd& X, const MatrixXd& y) {
    int n = X.rows(), correct = 0;
    for (int i = 0; i < n; i++) {
        auto p = predict(X.row(i).transpose());
        int pi = distance(p.data(), max_element(p.data(), p.data() + p.size()));
        int ri = distance(y.row(i).data(), max_element(y.row(i).data(), y.row(i).data() + y.row(i).size()));
        if (pi == ri) ++correct;
    }
    return double(correct) / n;
}

void PerceptronMulticapa::exportarErrores() {
    ofstream file("errores_por_epoca.csv");
    for (int i = 0; i < erroresPorEpoca.size(); ++i)
        file << i + 1 << "," << erroresPorEpoca[i] << "\n";
    file.close();
}

int main() {
    string csv = "mnist.csv";
    auto data = readCSV(csv);
    if (data.empty()) { cerr << "Datos vacíos"; return -1; }
    int total = data.size(), trainN = 10000;
    int feats = data[0].size() - 1;

    // Preparar training
    MatrixXd Xtr(trainN, feats);
    MatrixXd Ytr = MatrixXd::Zero(trainN, 10);
    for (int i = 0; i < trainN; i++) {
        for (int j = 1; j <= feats; j++) Xtr(i, j - 1) = data[i][j] / 255.0;
        Ytr(i, int(data[i][0])) = 1;
    }

    // Preparar test
    int testN = total - trainN;
    MatrixXd Xte(testN, feats);
    VectorXi Yte(testN);
    for (int i = trainN; i < total; i++) {
        for (int j = 1; j <= feats; j++) Xte(i - trainN, j - 1) = data[i][j] / 255.0;
        Yte(i - trainN) = int(data[i][0]);
    }

    // Entrenar y evaluar
    PerceptronMulticapa mlp(feats, 128, 10);
    cout << "Número de capas: 3 (entrada, oculta, salida)" << endl;
    cout << "Neuronas por capa: Entrada = " << feats << ", Oculta = 128, Salida = 10" << endl;

    mlp.train(Xtr, Ytr, 50, 0.01);

    int corr = 0;
    MatrixXi confusion = MatrixXi::Zero(10, 10);
    for (int i = 0; i < testN; i++) {
        auto p = mlp.predict(Xte.row(i).transpose());
        int pi = distance(p.data(), max_element(p.data(), p.data() + p.size()));
        confusion(Yte(i), pi)++;
        if (pi == Yte(i)) ++corr;
    }

    cout << "Precisión test: " << (double(corr) / testN) * 100 << "%\n";

    ofstream outConf("confusion_matrix.csv");
    outConf << "Predicted";
    for (int i = 0; i < 10; ++i) outConf << "," << i;
    outConf << "\n";
    for (int i = 0; i < 10; ++i) {
        outConf << i;
        for (int j = 0; j < 10; ++j) outConf << "," << confusion(i, j);
        outConf << "\n";
    }
    outConf.close();
    return 0;
}
