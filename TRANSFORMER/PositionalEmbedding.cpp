#include "PositionalEmbedding.hpp"

using namespace std;

PositionalEmbedding::PositionalEmbedding(int num_tokens, int dim, float threshold) {
    embeddings.resize(num_tokens, vector<float>(dim));
    for (auto& vec : embeddings) {
        for (auto& val : vec) {
            float raw = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
            val = ternarize(raw, threshold);  // ternarizado {–1, 0, +1}
        }
    }
}

float PositionalEmbedding::ternarize(float x, float threshold) const {
    if (x > threshold) return 1.0f;
    if (x < -threshold) return -1.0f;
    return 0.0f;
}

void PositionalEmbedding::apply(vector<vector<float>>& tokens) const {
    if (tokens.size() != embeddings.size() || tokens[0].size() != embeddings[0].size()) {
        throw runtime_error("Dimensiones incompatibles entre tokens y embeddings de posición.");
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        for (size_t j = 0; j < tokens[i].size(); ++j) {
            tokens[i][j] += embeddings[i][j];
        }
    }
}
