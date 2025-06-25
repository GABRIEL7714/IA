#pragma once
#include "BitAttention.hpp"
#include <stdexcept>
#include "BitMLP.hpp"

class BitTransformerEncoderLayer {
public:
    BitTransformerEncoderLayer(int dim, int hidden_dim = 128, float threshold = 0.33f);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& tokens);

    std::vector<std::vector<std::vector<float>>> backward(
    const std::vector<std::vector<float>>& grad_output);

    void update(const std::vector<std::vector<std::vector<float>>>& grads_attn,
            const std::vector<std::vector<std::vector<float>>>& grads_mlp,
            float lr);


private:
    BitAttention attention;
    BitMLP mlp;
    std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) const;

    std::vector<std::vector<float>> x_q_bin, x_k_bin, x_v_bin, x_o_bin;
    std::vector<std::vector<float>> x_mlp_bin;
    std::vector<std::vector<float>> residual;  // para skip connection

};
