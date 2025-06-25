#pragma once
#include <vector>

class BitLinearTrainable {
public:
BitLinearTrainable(int in_features, int out_features, float threshold = 0.33f);
// Forward binarizado: devuelve y = x_bin ⋅ W_binᵗ
std::vector<float> forward(const std::vector<float>& x, std::vector<float>& x_bin) const;

// Backward: devuelve ∂L/∂W_real ≈ ∂L/∂W_bin
std::vector<std::vector<float>> backward(const std::vector<float>& x_bin, const std::vector<float>& grad_output);
// Actualiza W_real y re-ternariza W_bin
void update(const std::vector<std::vector<float>>& grad, float lr);

private:
int in_features, out_features;
float threshold;
std::vector<std::vector<float>> W_real;
std::vector<std::vector<float>> W_bin;

float ternarize(float x) const;

};