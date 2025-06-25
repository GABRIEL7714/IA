#include "BitMLP.hpp"

BitMLP::BitMLP(int dim, int hidden_dim, float threshold)
    : fc1(dim, hidden_dim, threshold), fc2(hidden_dim, dim, threshold) {}

float BitMLP::relu(float x) const {
    return x > 0 ? x : 0;
}

std::vector<std::vector<float>> BitMLP::forward(
    const std::vector<std::vector<float>>& x,
    std::vector<std::vector<float>>& x_bin_store)
{
    const int N = x.size();
    std::vector<std::vector<float>> x_bin_fc1(N), x_bin_fc2(N);
    std::vector<std::vector<float>> out(N);

    for (int i = 0; i < N; ++i) {
        std::vector<float> h = fc1.forward(x[i], x_bin_fc1[i]);
        for (float& v : h)
            v = relu(v);
        out[i] = fc2.forward(h, x_bin_fc2[i]);
    }

    // Guardar binarizaciones de fc1 y fc2
    x_bin_store.clear();
    x_bin_store.insert(x_bin_store.end(), x_bin_fc1.begin(), x_bin_fc1.end());
    x_bin_store.insert(x_bin_store.end(), x_bin_fc2.begin(), x_bin_fc2.end());

    return out;
}
std::vector<std::vector<std::vector<float>>> BitMLP::backward(
    const std::vector<std::vector<float>>& x_bin_store,
    const std::vector<std::vector<float>>& grad_output)
{
    const int N = grad_output.size();

    std::vector<std::vector<std::vector<float>>> grads_fc1(N), grads_fc2(N);

    for (int i = 0; i < N; ++i) {
        // Dividir el binarizado total en 2 mitades: fc1 y fc2
        const auto& x_fc2 = x_bin_store[i + N];
        const auto& x_fc1 = x_bin_store[i];

        grads_fc2[i] = fc2.backward(x_fc2, grad_output[i]);
        grads_fc1[i] = fc1.backward(x_fc1, grad_output[i]); // simplificación
    }

    // Concatenar los dos conjuntos de gradientes
    grads_fc1.insert(grads_fc1.end(), grads_fc2.begin(), grads_fc2.end());
    return grads_fc1;
}

void BitMLP::update(const std::vector<std::vector<std::vector<float>>>& grads, float lr)
{
    const int N = grads.size() / 2;

    for (int i = 0; i < N; ++i)
        fc1.update(grads[i], lr);

    for (int i = N; i < 2 * N; ++i)
        fc2.update(grads[i], lr);
}
