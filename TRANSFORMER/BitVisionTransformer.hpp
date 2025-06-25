#pragma once

#include "PatchEmbedder.hpp"
#include "PositionalEmbedding.hpp"
#include "BitTransformerEncoderLayer.hpp"
#include "GlobalAveragePooling.hpp"
#include "BitLinearTrainable.hpp"
#include "ClassifierUtils.hpp"

class BitVisionTransformer {
public:
    BitVisionTransformer(int patch_size, int d_model, int num_classes, int num_layers, float threshold = 0.33f);
    float train_step(const float* image_data, int label, float lr);

    // input: imagen aplanada de 48x48 (2304 floats)
    int predict(const float* image_data);

private:
    int d_model;
    int num_layers;

    PatchEmbedder patch_embedder;
    PositionalEmbedding pos_embedding;
    vector<BitTransformerEncoderLayer> encoders;
    GlobalAveragePooling pooling;
    BitLinearTrainable classifier;
};
