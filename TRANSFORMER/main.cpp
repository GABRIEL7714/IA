#include "BitVisionTransformer.hpp"
#include <fstream>
#include <iostream>

int main() {
    // Cargar imagen bin (1 imagen de 48x48)
    std::ifstream in("prePros/X_test.bin", std::ios::binary);
    float image[48 * 48];
    in.read(reinterpret_cast<char*>(image), sizeof(image));
    in.close();

    // Crear el modelo completo
    BitVisionTransformer model(6, 64, 7, 2);  // patch 6x6, d_model=64, 7 clases, 2 capas encoder

    // Ejecutar predicción
    int predicted = model.predict(image);
    cout << "Emoción predicha: " << predicted << "\n";

    return 0;
}
