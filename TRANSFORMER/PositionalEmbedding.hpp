#pragma once
#include <vector>
#include <cstdlib>
#include <cmath>
#include <stdexcept>


using namespace std;

class PositionalEmbedding {
public:
    PositionalEmbedding(int num_tokens, int dim, float threshold = 0.33f);

    void apply(vector<vector<float>>& tokens) const;

private:
    vector<vector<float>> embeddings;

    float ternarize(float x, float threshold = 0.33f) const;
};
