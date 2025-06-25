#include "BitVisionTransformer.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>

using namespace std;

constexpr int IMAGE_SIZE = 48 * 48; // 2304
constexpr int NUM_CLASSES = 7;

// Función para cargar X e Y desde archivos binarios
bool load_data(const string& x_path, const string& y_path,
               vector<vector<float>>& X, vector<int>& Y) {
    ifstream fx(x_path, ios::binary);
    ifstream fy(y_path, ios::binary);

    if (!fx || !fy) {
        cerr << "Error al abrir los archivos binarios.\n";
        return false;
    }

    fx.seekg(0, ios::end);
    size_t x_size = fx.tellg();
    fx.seekg(0, ios::beg);
    size_t num_images = x_size / (IMAGE_SIZE * sizeof(float));

    X.resize(num_images, vector<float>(IMAGE_SIZE));
    Y.resize(num_images);

    for (size_t i = 0; i < num_images; ++i)
        fx.read(reinterpret_cast<char*>(X[i].data()), IMAGE_SIZE * sizeof(float));

    vector<uint8_t> raw_labels(num_images);
    fy.read(reinterpret_cast<char*>(raw_labels.data()), num_images);
    for (size_t i = 0; i < num_images; ++i)
        Y[i] = static_cast<int>(raw_labels[i]);

    fx.close();
    fy.close();

    return true;
}

int main() {
    vector<vector<float>> X_train;
    vector<int> Y_train;

    if (!load_data("prePros/X_train.bin", "prePros/Y_train.bin", X_train, Y_train))
        return 1;

    cout << "Datos cargados: " << X_train.size() << " imágenes\n";
    cout << "Comenzando entrenamiento...\n";

    int patch_size = 6;
    int d_model = 64;
    int num_layers = 2;
    float threshold = 0.33f;
    float lr = 0.01f;
    int epochs = 5;

    BitVisionTransformer model(patch_size, d_model, NUM_CLASSES, num_layers, threshold);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < X_train.size(); ++i) {
            float loss = model.train_step(X_train[i].data(), Y_train[i], lr);
            total_loss += loss;

            int pred = model.predict(X_train[i].data());
            if (pred == Y_train[i]) ++correct;

            if (i % 1000 == 0)
                cout << "[Epoca " << epoch + 1 << "] Imagen " << i
                     << " | Perdida: " << loss
                     << " | Precision parcial: " << (100.0f * correct / (i + 1)) << "%\n";
        }

        float avg_loss = total_loss / X_train.size();
        float accuracy = 100.0f * correct / X_train.size();

        cout << "Epoca " << epoch + 1
             << " - Perdida promedio: " << avg_loss
             << " | Precision: " << accuracy << "%\n";
    }

    cout << "Entrenamiento finalizado. Presiona ENTER para salir...\n";
    cin.get();
    cin.get();

    return 0;
}
