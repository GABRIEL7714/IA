#include "BitTransformerEncoderLayer.hpp"
#include <stdexcept>


BitTransformerEncoderLayer::BitTransformerEncoderLayer(int dim, int hidden_dim, float threshold)
    : attention(dim, threshold), mlp(dim, hidden_dim, threshold) {}

std::vector<std::vector<float>> BitTransformerEncoderLayer::forward(
    const std::vector<std::vector<float>>& tokens)
{
    residual = tokens;
    x_q_bin.clear(); x_k_bin.clear(); x_v_bin.clear(); x_o_bin.clear();

    auto attended = attention.forward(tokens, x_q_bin, x_k_bin, x_v_bin, x_o_bin);

    std::vector<std::vector<float>> normed = attended;  // sin LayerNorm aún
    std::vector<std::vector<float>> added(normed.size());

    for (size_t i = 0; i < normed.size(); ++i) {
        added[i].resize(normed[i].size());
        for (size_t j = 0; j < normed[i].size(); ++j)
            added[i][j] = normed[i][j] + residual[i][j];  // skip connection
    }

    return mlp.forward(added, x_mlp_bin);  // guarda binarizado también
}

std::vector<std::vector<std::vector<float>>> BitTransformerEncoderLayer::backward(
    const std::vector<std::vector<float>>& grad_output)
{
    auto grads_mlp = mlp.backward(x_mlp_bin, grad_output);

std::vector<std::vector<float>> grad_output_attn;
grad_output_attn.reserve(residual.size());

for (size_t i = 0; i < residual.size(); ++i)
    grad_output_attn.push_back(grads_mlp[i][0]);

auto grads_attn = attention.backward(
    grad_output_attn, x_q_bin, x_k_bin, x_v_bin, x_o_bin);

}


void BitTransformerEncoderLayer::update(
    const std::vector<std::vector<std::vector<float>>>& grads_attn,
    const std::vector<std::vector<std::vector<float>>>& grads_mlp,
    float lr)
{
    attention.update(grads_attn, grads_attn, grads_attn, grads_attn, lr);  // usando el mismo por ahora
    mlp.update(grads_mlp, lr);
}


std::vector<float> BitTransformerEncoderLayer::add(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.size() != b.size())
        throw std::runtime_error("Dimensiones incompatibles en residual.");
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}
