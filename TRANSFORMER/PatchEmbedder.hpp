#pragma once
#include <vector>
#include "BitLinearTrainable.hpp"

class PatchEmbedder {
public:
PatchEmbedder(int patch_size, int embed_dim, float threshold = 0.33f);
std::vector<std::vector<float>> extract_patches(const float* image48x48) const;
std::vector<std::vector<float>> embed_patches(const std::vector<std::vector<float>>& patches,
                                              std::vector<std::vector<float>>& x_bin_store) const;
std::vector<std::vector<float>> process(const float* image48x48, std::vector<std::vector<float>>& x_bin_store);

// Nuevos métodos para entrenamiento
std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<float>>& grad_output,
                                                      const std::vector<std::vector<float>>& x_bin_store);

void update(const std::vector<std::vector<std::vector<float>>>& grads, float lr);
private:
int patch_size;
int embed_dim;
int input_dim;
BitLinearTrainable linear;

};