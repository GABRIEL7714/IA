#pragma once
#include "BitLinearTrainable.hpp"
#include <vector>

class BitMLP {
public:
    BitMLP(int dim, int hidden_dim, float threshold = 0.33f);

    std::vector<std::vector<float>> forward(
    const std::vector<std::vector<float>>& x,
    std::vector<std::vector<float>>& x_bin_store);
    std::vector<std::vector<std::vector<float>>> backward(
    const std::vector<std::vector<float>>& x_bin_store,
    const std::vector<std::vector<float>>& grad_output);

void update(const std::vector<std::vector<std::vector<float>>>& grads, float lr);


private:
    BitLinearTrainable fc1;
    BitLinearTrainable fc2;

    float relu(float x) const;
};
