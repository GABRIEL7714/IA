#include "PatchEmbedder.hpp"
#include <cmath>
#include <cstdlib>

PatchEmbedder::PatchEmbedder(int patch_size, int embed_dim, float threshold)
: patch_size(patch_size),
embed_dim(embed_dim),
input_dim(patch_size * patch_size),
linear(input_dim, embed_dim, threshold) {}

std::vector<std::vector<float>> PatchEmbedder::extract_patches(const float* image48x48) const {
const int image_size = 48;
const int filas = image_size / patch_size;
std::vector<std::vector<float>> patches;
patches.reserve(filas * filas);
for (int y = 0; y < image_size; y += patch_size) {
    for (int x = 0; x < image_size; x += patch_size) {
        std::vector<float> patch(input_dim);
        for (int dy = 0; dy < patch_size; ++dy) {
            for (int dx = 0; dx < patch_size; ++dx) {
                int idx = (y + dy) * image_size + (x + dx);
                patch[dy * patch_size + dx] = image48x48[idx];
            }
        }
        patches.push_back(patch);
    }
}

return patches;}
std::vector<std::vector<float>> PatchEmbedder::embed_patches(const std::vector<std::vector<float>>& patches,
std::vector<std::vector<float>>& x_bin_store) const {
std::vector<std::vector<float>> embeddings;
embeddings.reserve(patches.size());
x_bin_store.resize(patches.size());
for (size_t i = 0; i < patches.size(); ++i) {
    embeddings.push_back(linear.forward(patches[i], x_bin_store[i]));
}

return embeddings;}
std::vector<std::vector<float>> PatchEmbedder::process(const float* image48x48,
std::vector<std::vector<float>>& x_bin_store) {
auto patches = extract_patches(image48x48);
return embed_patches(patches, x_bin_store);
}

std::vector<std::vector<std::vector<float>>> PatchEmbedder::backward(const std::vector<std::vector<float>>& grad_output,
const std::vector<std::vector<float>>& x_bin_store) {
std::vector<std::vector<std::vector<float>>> grads(grad_output.size());
for (size_t i = 0; i < grad_output.size(); ++i)
grads[i] = linear.backward(x_bin_store[i], grad_output[i]);
return grads;
}

void PatchEmbedder::update(const std::vector<std::vector<std::vector<float>>>& grads, float lr) {
for (const auto& g : grads)
linear.update(g, lr);
}