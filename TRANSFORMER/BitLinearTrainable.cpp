#include "BitLinearTrainable.hpp"
#include <cstdlib>
#include <cmath>

BitLinearTrainable::BitLinearTrainable(int in_f, int out_f, float thresh)
: in_features(in_f), out_features(out_f), threshold(thresh) {
W_real.resize(out_features, std::vector<float>(in_features));
W_bin.resize(out_features, std::vector<float>(in_features));
for (int i = 0; i < out_features; ++i) {
for (int j = 0; j < in_features; ++j) {
float r = ((rand() / (float)RAND_MAX) * 2.0f) - 1.0f;
W_real[i][j] = r;
W_bin[i][j] = ternarize(r);
}
}
}

float BitLinearTrainable::ternarize(float x) const {
if (x > threshold) return 1.0f;
if (x < -threshold) return -1.0f;
return 0.0f;
}

std::vector<float> BitLinearTrainable::forward(const std::vector<float>& x, std::vector<float>& x_bin) const {
x_bin.resize(in_features);
for (int i = 0; i < in_features; ++i)
x_bin[i] = ternarize(x[i]);
std::vector<float> output(out_features, 0.0f);
for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j)
        output[i] += x_bin[j] * W_bin[i][j];
}
return output;}
std::vector<std::vector<float>> BitLinearTrainable::backward(const std::vector<float>& x_bin,
const std::vector<float>& grad_output) {
std::vector<std::vector<float>> grad(out_features, std::vector<float>(in_features));
for (int i = 0; i < out_features; ++i) {
for (int j = 0; j < in_features; ++j) {
grad[i][j] = grad_output[i] * x_bin[j]; // ∂L/∂W_bin_ij ≈ ∂L/∂W_real_ij
}
}
return grad;
}

void BitLinearTrainable::update(const std::vector<std::vector<float>>& grad, float lr) {
for (int i = 0; i < out_features; ++i) {
for (int j = 0; j < in_features; ++j) {
W_real[i][j] -= lr * grad[i][j];
W_bin[i][j] = ternarize(W_real[i][j]); // actualizar W_bin después del paso
}
}
}