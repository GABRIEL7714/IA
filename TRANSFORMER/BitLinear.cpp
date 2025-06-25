#include "BitLinear.hpp"
#include <cstdlib>
#include <stdexcept>
#include <cmath>

using namespace std;

BitLinear::BitLinear(int in_features, int out_features, float threshold)
    : in_features(in_features), out_features(out_features) {
    weights.resize(out_features, vector<float>(in_features));
    initialize_weights(threshold);
}

void BitLinear::initialize_weights(float threshold) {
    for (auto& row : weights) {
        for (auto& w : row) {
            float raw = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
            w = ternarize(raw, threshold);
        }
    }
}

float BitLinear::ternarize(float x, float threshold) const {
    if (x > threshold) return 1.0f;
    if (x < -threshold) return -1.0f;
    return 0.0f;
}

vector<float> BitLinear::forward(const vector<float>& input) const {
    if ((int)input.size() != in_features)
        throw runtime_error("Dimensión de entrada incorrecta en BitLinear");

    vector<float> output(out_features, 0.0f);
    vector<float> bin_input(in_features);

    // Ternarizar entrada
    for (int i = 0; i < in_features; ++i)
        bin_input[i] = ternarize(input[i]);

    // Multiplicación binaria: dot product entre bin_input y cada fila de weights
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j)
            output[i] += bin_input[j] * weights[i][j];
    }

    return output;
}
