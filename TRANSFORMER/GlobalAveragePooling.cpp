#include "GlobalAveragePooling.hpp"

vector<float> GlobalAveragePooling::forward(const vector<vector<float>>& tokens) const {
    int n_tokens = tokens.size();
    int dim = tokens[0].size();

    vector<float> pooled(dim, 0.0f);

    for (const auto& token : tokens) {
        for (int i = 0; i < dim; ++i)
            pooled[i] += token[i];
    }

    for (int i = 0; i < dim; ++i)
        pooled[i] /= n_tokens;

    return pooled;
}
