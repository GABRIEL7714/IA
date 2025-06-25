#pragma once
#include "BitLinearTrainable.hpp"
#include <vector>

class BitAttention {
public:
    BitAttention(int dim, float threshold = 0.33f);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& x,
        std::vector<std::vector<float>>& x_q_bin_store,
        std::vector<std::vector<float>>& x_k_bin_store,
        std::vector<std::vector<float>>& x_v_bin_store,
        std::vector<std::vector<float>>& x_o_bin_store) const;

    std::vector<std::vector<std::vector<float>>> backward(
        const std::vector<std::vector<float>>& grad_output,
        const std::vector<std::vector<float>>& x_q_bin_store,
        const std::vector<std::vector<float>>& x_k_bin_store,
        const std::vector<std::vector<float>>& x_v_bin_store,
        const std::vector<std::vector<float>>& x_o_bin_store);

    void update(const std::vector<std::vector<std::vector<float>>>& grads_q,
                const std::vector<std::vector<std::vector<float>>>& grads_k,
                const std::vector<std::vector<std::vector<float>>>& grads_v,
                const std::vector<std::vector<std::vector<float>>>& grads_o,
                float lr);

private:
    BitLinearTrainable q_proj;
    BitLinearTrainable k_proj;
    BitLinearTrainable v_proj;
    BitLinearTrainable o_proj;
};
