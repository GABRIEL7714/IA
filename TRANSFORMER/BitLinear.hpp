#pragma once
#include <vector>

using namespace std;

class BitLinear {
public:
    BitLinear(int in_features, int out_features, float threshold = 0.33f);

    vector<float> forward(const vector<float>& input) const;
    void initialize_weights(float threshold = 0.33f);

private:
    int in_features;
    int out_features;
    vector<vector<float>> weights;

    float ternarize(float x, float threshold = 0.33333f) const;
};
