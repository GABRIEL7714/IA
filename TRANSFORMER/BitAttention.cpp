#include "BitAttention.hpp"
#include <cmath>
#include <algorithm>

BitAttention::BitAttention(int dim, float threshold)
    : q_proj(dim, dim, threshold),
      k_proj(dim, dim, threshold),
      v_proj(dim, dim, threshold),
      o_proj(dim, dim, threshold) {}

std::vector<std::vector<float>> BitAttention::forward(
    const std::vector<std::vector<float>>& x,
    std::vector<std::vector<float>>& x_q_bin_store,
    std::vector<std::vector<float>>& x_k_bin_store,
    std::vector<std::vector<float>>& x_v_bin_store,
    std::vector<std::vector<float>>& x_o_bin_store) const
{
    const int N = x.size();
    std::vector<std::vector<float>> Q(N), K(N), V(N);

    x_q_bin_store.resize(N);
    x_k_bin_store.resize(N);
    x_v_bin_store.resize(N);
    x_o_bin_store.resize(N);

    for (int i = 0; i < N; ++i) {
        Q[i] = q_proj.forward(x[i], x_q_bin_store[i]);
        K[i] = k_proj.forward(x[i], x_k_bin_store[i]);
        V[i] = v_proj.forward(x[i], x_v_bin_store[i]);
    }

    // Escalar y calcular atención (QKᵗ)
    std::vector<std::vector<float>> attn(N, std::vector<float>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (size_t d = 0; d < Q[i].size(); ++d)
                attn[i][j] += Q[i][d] * K[j][d];

    // Softmax fila por fila
    for (int i = 0; i < N; ++i) {
        float max_val = *max_element(attn[i].begin(), attn[i].end());
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn[i][j] = std::exp(attn[i][j] - max_val);
            sum += attn[i][j];
        }
        for (int j = 0; j < N; ++j)
            attn[i][j] /= sum;
    }

    // Atención × V
    std::vector<std::vector<float>> weighted(N, std::vector<float>(V[0].size(), 0.0f));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (size_t d = 0; d < V[0].size(); ++d)
                weighted[i][d] += attn[i][j] * V[j][d];

    // Proyección final con W_o
    std::vector<std::vector<float>> output(N);
    for (int i = 0; i < N; ++i)
        output[i] = o_proj.forward(weighted[i], x_o_bin_store[i]);

    return output;
}
std::vector<std::vector<std::vector<float>>> BitAttention::backward(
    const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& x_q_bin_store,
    const std::vector<std::vector<float>>& x_k_bin_store,
    const std::vector<std::vector<float>>& x_v_bin_store,
    const std::vector<std::vector<float>>& x_o_bin_store)
{
    const int N = grad_output.size();
    std::vector<std::vector<std::vector<float>>> grads_o(N);

    // backward solo para la proyección final (por ahora)
    for (int i = 0; i < N; ++i) {
        grads_o[i] = o_proj.backward(x_o_bin_store[i], grad_output[i]);
    }

    // devolvemos solo los grads_o, los demás los puedes ignorar si aún no están implementados
    return grads_o;
}
void BitAttention::update(const std::vector<std::vector<std::vector<float>>>& grads_q,
                          const std::vector<std::vector<std::vector<float>>>& grads_k,
                          const std::vector<std::vector<std::vector<float>>>& grads_v,
                          const std::vector<std::vector<std::vector<float>>>& grads_o,
                          float lr)
{
    // (opcional: si en el futuro agregas grads_q, grads_k, grads_v)

    for (const auto& g : grads_q) q_proj.update(g, lr);
    for (const auto& g : grads_k) k_proj.update(g, lr);
    for (const auto& g : grads_v) v_proj.update(g, lr);
    for (const auto& g : grads_o) o_proj.update(g, lr);
}
