#include "ClassifierUtils.hpp"

int argmax(const vector<float>& logits) {
    int max_idx = 0;
    float max_val = logits[0];

    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    return max_idx;
}
