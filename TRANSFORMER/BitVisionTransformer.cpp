#include "BitVisionTransformer.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

static std::vector<float> softmax(const std::vector<float>& logits) {
float max_logit = *std::max_element(logits.begin(), logits.end());
std::vector<float> exps(logits.size());
float sum = 0.0f;
for (size_t i = 0; i < logits.size(); ++i) {
exps[i] = std::exp(logits[i] - max_logit);
sum += exps[i];
}
for (float& val : exps) val /= sum;
return exps;
}

float BitVisionTransformer::train_step(const float* image_data, int label, float lr) {
    //std::cout << "train_step con label = " << label << "\n";

std::vector<std::vector<float>> x_bin_store;
// 1. Imagen → parches → embeddings
std::vector<std::vector<float>> tokens = patch_embedder.process(image_data, x_bin_store);

// 2. Positional embedding
pos_embedding.apply(tokens);

// 3. Encoder (aún fijo)
for (int i = 0; i < num_layers; ++i)
    tokens = encoders[i].forward(tokens);

// 4. Global Average Pooling
std::vector<float> pooled = pooling.forward(tokens);

// 5. Clasificación
std::vector<float> x_bin_clf;
std::vector<float> logits = classifier.forward(pooled, x_bin_clf);

// 6. Pérdida (cross entropy)
std::vector<float> probs = softmax(logits);
float loss = -std::log(probs[label] + 1e-8f);
std::vector<float> grad_logits = probs;
grad_logits[label] -= 1.0f;

// 7. Backprop del clasificador
auto grad_clf = classifier.backward(x_bin_clf, grad_logits);
classifier.update(grad_clf, lr);

// 8. Simular gradiente para patch_embedder
std::vector<std::vector<float>> grad_tokens(tokens.size(), std::vector<float>(tokens[0].size(), 1.0f));
auto grad_patch = patch_embedder.backward(grad_tokens, x_bin_store);
patch_embedder.update(grad_patch, lr);

return loss;
}
BitVisionTransformer::BitVisionTransformer(int patch_size, int d_model, int num_classes, int num_layers, float threshold)
    : d_model(d_model),
      num_layers(num_layers),
      patch_embedder(patch_size, d_model),
      pos_embedding(64, d_model, threshold),
      classifier(d_model, num_classes, threshold) {
    
    for (int i = 0; i < num_layers; ++i)
        encoders.emplace_back(BitTransformerEncoderLayer(d_model, d_model * 2, threshold));
}

int BitVisionTransformer::predict(const float* image_data) {
    // Paso 1: patching y embedding
    std::vector<std::vector<float>> x_bin_store;
    vector<vector<float>> tokens = patch_embedder.process(image_data,x_bin_store);

    // Paso 2: agregar positional embedding
    pos_embedding.apply(tokens);

    // Paso 3: encoder layers
    for (int i = 0; i < num_layers; ++i)
        tokens = encoders[i].forward(tokens);

    // Paso 4: Global Average Pooling
    vector<float> pooled = pooling.forward(tokens);

    // Paso 5: Clasificación
    vector<float> x_bin_dummy;
    vector<float> logits = classifier.forward(pooled, x_bin_dummy);
    // Paso 6: Predicción final
    return argmax(logits);
}
