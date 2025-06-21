#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct Dataset {
    vector<vector<float>> inputs;
    vector<vector<float>> outputs;
};

__global__ void forward_kernel(
    const float* input, const float* weights, const float* bias,
    float* output, int in_size, int out_size, int batch_size, bool use_relu
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx >= batch_size || neuron_idx >= out_size) return;

    float sum = 0.0f;
    for (int k = 0; k < in_size; ++k) {
        sum += input[sample_idx * in_size + k] * weights[neuron_idx * in_size + k];
    }
    sum += bias[neuron_idx];

    if (use_relu) {
        sum = fmaxf(0.0f, sum);
    }

    output[sample_idx * out_size + neuron_idx] = sum;
}

__global__ void output_error_kernel(
    const float* output,
    const float* target,
    float* error,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Strict bounds checking
    if (idx >= total_elements) return;

    // Safe error calculation
    error[idx] = output[idx] - target[idx];
}

__global__ void hidden_error_kernel(
    const float* activation,
    const float* current_error,
    const float* current_weights,
    float* prev_error,
    int current_size,
    int next_size,
    int batch_size
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx >= batch_size || neuron_idx >= current_size) return;

    float sum = 0.0f;
    for (int k = 0; k < next_size; ++k) {
        int weight_idx = neuron_idx * next_size + k;
        int error_idx = sample_idx * next_size + k;

        if (weight_idx < current_size * next_size && error_idx < batch_size * next_size) {
            sum += current_weights[weight_idx] * current_error[error_idx];
        }
    }

    int output_idx = sample_idx * current_size + neuron_idx;
    if (output_idx < batch_size * current_size) {
        float derivative = (activation[output_idx] > 0.0f) ? 1.0f : 0.0f;
        prev_error[output_idx] = sum * derivative;
    }
}

__global__ void update_weights_kernel(
    float* weights,
    float* bias,
    const float* input,
    const float* error,
    float lr,
    int in_size,
    int out_size,
    int batch_size
) {
    int neuron_out = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_in = blockIdx.y * blockDim.y + threadIdx.y;

    if (neuron_out < out_size && neuron_in < in_size) {
        float grad = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad += error[i * out_size + neuron_out] * input[i * in_size + neuron_in];
        }
        atomicAdd(&weights[neuron_out * in_size + neuron_in], -lr * grad / batch_size);
    }

    if (neuron_out < out_size && neuron_in == 0) {
        float grad_b = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad_b += error[i * out_size + neuron_out];
        }
        atomicAdd(&bias[neuron_out], -lr * grad_b / batch_size);
    }
}

Dataset loadCSV(const string& filename, size_t max_samples) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    size_t count = 0;
    while (getline(file, line) && count < max_samples) {
        stringstream ss(line);
        string value;

        vector<float> input(784);
        vector<float> output(10, 0.0f);

        getline(ss, value, ',');
        int label = stoi(value);
        output[label] = 1.0f;

        for (int i = 0; i < 784; ++i) {
            if (!getline(ss, value, ',')) break;
            input[i] = stof(value) / 255.0f;
        }

        data.inputs.push_back(input);
        data.outputs.push_back(output);
        count++;
    }
    cout << "Loaded " << data.inputs.size() << " samples." << endl;
    return data;
}

struct MLP {
    int num_layers;
    vector<int> layer_sizes;
    float learning_rate;
    int batch_size;

    float** d_weights;
    float** d_biases;
    float** d_outputs;
    float** d_errors;

    void init(const vector<int>& sizes, float lr, int batch) {
        // Validate parameters
        if (sizes.size() < 2 || lr <= 0 || batch <= 0) {
            throw invalid_argument("Invalid MLP initialization parameters");
        }

        num_layers = sizes.size();
        layer_sizes = sizes;
        learning_rate = lr;
        batch_size = batch;

        // Allocate pointer arrays
        d_weights = new float* [num_layers - 1]();
        d_biases = new float* [num_layers - 1]();
        d_outputs = new float* [num_layers]();
        d_errors = new float* [num_layers - 1]();

        // Allocate GPU memory with verification
        for (int i = 0; i < num_layers; ++i) {
            CHECK_CUDA(cudaMalloc(&d_outputs[i], sizeof(float) * layer_sizes[i] * batch_size));
            //cout << "Allocated d_outputs[" << i << "] with size " << layer_sizes[i] * batch_size << endl;
        }

        for (int i = 0; i < num_layers - 1; ++i) {
            int in_size = layer_sizes[i];
            int out_size = layer_sizes[i + 1];

            CHECK_CUDA(cudaMalloc(&d_weights[i], sizeof(float) * in_size * out_size));
            CHECK_CUDA(cudaMalloc(&d_biases[i], sizeof(float) * out_size));
            CHECK_CUDA(cudaMalloc(&d_errors[i], sizeof(float) * out_size * batch_size));

            //cout << "Allocated layer " << i << " weights: " << in_size << "x" << out_size
                //<< ", errors: " << out_size * batch_size << endl;

            // Initialize weights and biases
            float* h_weights = new float[in_size * out_size];
            for (int j = 0; j < in_size * out_size; ++j) {
                h_weights[j] = (rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            }
            CHECK_CUDA(cudaMemcpy(d_weights[i], h_weights, sizeof(float) * in_size * out_size, cudaMemcpyHostToDevice));
            delete[] h_weights;

            float* h_biases = new float[out_size];
            for (int j = 0; j < out_size; ++j) {
                h_biases[j] = (rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            }
            CHECK_CUDA(cudaMemcpy(d_biases[i], h_biases, sizeof(float) * out_size, cudaMemcpyHostToDevice));
            delete[] h_biases;
        }
    }

    void free() {
        for (int i = 0; i < num_layers - 1; ++i) {
            if (d_weights[i]) cudaFree(d_weights[i]);
            if (d_biases[i]) cudaFree(d_biases[i]);
            if (d_errors[i]) cudaFree(d_errors[i]);
        }
        for (int i = 0; i < num_layers; ++i) {
            if (d_outputs[i]) cudaFree(d_outputs[i]);
        }

        delete[] d_weights;
        delete[] d_biases;
        delete[] d_outputs;
        delete[] d_errors;
    }

    void forward(float* d_input, int current_batch_size) {
        // Validate input
        if (current_batch_size <= 0 || current_batch_size > batch_size) {
            throw invalid_argument("Invalid batch size in forward()");
        }

        // Copy input to first layer
        CHECK_CUDA(cudaMemcpy(d_outputs[0], d_input, sizeof(float) * current_batch_size * layer_sizes[0], cudaMemcpyDeviceToDevice));

        // Forward pass through layers
        for (int l = 0; l < num_layers - 1; ++l) {
            int in_size = layer_sizes[l];
            int out_size = layer_sizes[l + 1];
            bool is_output_layer = (l == num_layers - 2);

            dim3 block(16, 16);
            dim3 grid(
                (current_batch_size + block.x - 1) / block.x,
                (out_size + block.y - 1) / block.y
            );

            //cout << "Forward layer " << l << " grid: (" << grid.x << ", " << grid.y << ")" << endl;

            forward_kernel << <grid, block >> > (
                d_outputs[l],
                d_weights[l],
                d_biases[l],
                d_outputs[l + 1],
                in_size,
                out_size,
                current_batch_size,
                !is_output_layer
                );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void backward(float* d_input, float* d_target, int current_batch_size) {
        // Validate input
        if (current_batch_size <= 0 || current_batch_size > batch_size) {
            throw invalid_argument("Invalid batch size in backward()");
        }

        // 1. Calculate output layer error
        int output_layer_idx = num_layers - 1;
        int output_size = layer_sizes[output_layer_idx];
        int total_output = current_batch_size * output_size;

        if (output_layer_idx - 1 < 0 || output_layer_idx - 1 >= num_layers - 1) {
            throw runtime_error("Invalid layer index in backward()");
        }

        //cout << "Calculating output error with " << total_output << " elements" << endl;

        int threadsPerBlock = 256;
        int blocksPerGrid = (total_output + threadsPerBlock - 1) / threadsPerBlock;

        output_error_kernel << <blocksPerGrid, threadsPerBlock >> > (
            d_outputs[output_layer_idx],
            d_target,
            d_errors[output_layer_idx - 1],
            total_output
            );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Backpropagate through hidden layers
        for (int l = num_layers - 2; l >= 1; --l) {
            int current_size = layer_sizes[l];
            int next_size = layer_sizes[l + 1];

            dim3 block(16, 16);
            dim3 grid(
                (current_batch_size + block.x - 1) / block.x,
                (current_size + block.y - 1) / block.y
            );

            cout << "Backward layer " << l << " grid: (" << grid.x << ", " << grid.y << ")" << endl;

            hidden_error_kernel << <grid, block >> > (
                d_outputs[l],
                d_errors[l],
                d_weights[l],
                d_errors[l - 1],
                current_size,
                next_size,
                current_batch_size
                );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // 3. Update weights and biases
        for (int l = 0; l < num_layers - 1; ++l) {
            int in_size = layer_sizes[l];
            int out_size = layer_sizes[l + 1];

            dim3 block(16, 16);
            dim3 grid(
                (out_size + block.x - 1) / block.x,
                (in_size + block.y - 1) / block.y
            );

            const float* input = (l == 0) ? d_input : d_outputs[l];

            update_weights_kernel << <grid, block >> > (
                d_weights[l],
                d_biases[l],
                input,
                d_errors[l],
                learning_rate,
                in_size,
                out_size,
                current_batch_size
                );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    float compute_error(float* d_target, int current_batch_size) {
        int output_size = layer_sizes.back();
        int total = current_batch_size * output_size;

        float* d_error;
        CHECK_CUDA(cudaMalloc(&d_error, sizeof(float) * total));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        output_error_kernel << <blocks, threads >> > (
            d_outputs[num_layers - 1],
            d_target,
            d_error,
            total
            );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        vector<float> h_error(total);
        CHECK_CUDA(cudaMemcpy(h_error.data(), d_error, sizeof(float) * total, cudaMemcpyDeviceToHost));

        float sum = 0.0f;
        for (float val : h_error) sum += val * val;

        cudaFree(d_error);
        return sum / (2 * current_batch_size);
    }

    void log_error(int epoch, float error, const string& filename) {
        ofstream file(filename, ios::app);
        file << epoch << "," << error << "\n";
        file.close();
    }
};



int main() {
    try {
        // Configuración
        string csv_path = "mnist.csv";
        int total_samples = 60000;
        int batch_size = 60000;
        float learning_rate = 0.01f;
        int epochs = 100;
        vector<int> architecture = { 784, 128, 10 };  // 1 capa oculta con 128 neuronas
        // Cargar datos
        cout << "Loading dataset..." << endl;
        Dataset dataset = loadCSV(csv_path, total_samples);
        int num_batches = total_samples / batch_size;

        // Preparar datos en GPU
        cout << "Preparing GPU data..." << endl;
        float* h_inputs = new float[total_samples * 784];
        float* h_outputs = new float[total_samples * 10];

        for (int i = 0; i < total_samples; ++i) {
            copy(dataset.inputs[i].begin(), dataset.inputs[i].end(), h_inputs + i * 784);
            copy(dataset.outputs[i].begin(), dataset.outputs[i].end(), h_outputs + i * 10);
        }

        float* d_inputs, * d_outputs;
        CHECK_CUDA(cudaMalloc(&d_inputs, sizeof(float) * total_samples * 784));
        CHECK_CUDA(cudaMalloc(&d_outputs, sizeof(float) * total_samples * 10));
        CHECK_CUDA(cudaMemcpy(d_inputs, h_inputs, sizeof(float) * total_samples * 784, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_outputs, h_outputs, sizeof(float) * total_samples * 10, cudaMemcpyHostToDevice));

        delete[] h_inputs;
        delete[] h_outputs;

        // Inicializar red
        cout << "Initializing MLP..." << endl;
        MLP net;
        net.init(architecture, learning_rate, batch_size);

        // Entrenamiento
        cout << "Training..." << endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_error = 0.0f;

            for (int i = 0; i < num_batches; ++i) {
                int input_offset = i * batch_size * 784;  // Para imágenes (784)
                int output_offset = i * batch_size * 10;  // Para etiquetas (10)

                float* d_batch_in = d_inputs + input_offset;
                float* d_batch_out = d_outputs + output_offset;

                cout << "Epoch " << epoch << ", Batch " << i << endl;

                net.forward(d_batch_in, batch_size);
                net.backward(d_batch_in, d_batch_out, batch_size);

                float batch_error = net.compute_error(d_batch_out, batch_size);
                total_error += batch_error;

                cout << "Batch error: " << batch_error << endl;
            }

            float avg_error = total_error / num_batches;
            net.log_error(epoch, avg_error, "training_error.csv");
            cout << "Epoch " << epoch << " - Avg Error: " << avg_error << endl;
        }

        // Liberar memoria
        cout << "Cleaning up..." << endl;
        net.free();
        CHECK_CUDA(cudaFree(d_inputs));
        CHECK_CUDA(cudaFree(d_outputs));

        cout << "Training complete!" << endl;

        // --- Evaluación Final (después del entrenamiento) ---
        if (total_samples >= 10000) {
            cout << "\nEvaluando primeros 10,000 ejemplos..." << endl;

            // 1. Matriz de confusión (10x10) inicializada en ceros
            vector<vector<int>> confusion_matrix(10, vector<int>(10, 0));

            // 2. Procesar los primeros 10,000 datos
            int test_size = 10000;
            float* d_test_input;
            CHECK_CUDA(cudaMalloc(&d_test_input, sizeof(float) * test_size * 784));
            CHECK_CUDA(cudaMemcpy(d_test_input, d_inputs, sizeof(float) * test_size * 784, cudaMemcpyDeviceToDevice));

            // 3. Forward pass para obtener predicciones
            net.forward(d_test_input, test_size);

            // 4. Copiar salidas a CPU
            float* h_predictions = new float[test_size * 10];
            CHECK_CUDA(cudaMemcpy(h_predictions, net.d_outputs[net.num_layers - 1], sizeof(float) * test_size * 10, cudaMemcpyDeviceToHost));

            // 5. Generar matriz de confusión
            for (int i = 0; i < test_size; ++i) {
                // Obtener etiqueta real (del CSV)
                int true_label = distance(dataset.outputs[i].begin(), max_element(dataset.outputs[i].begin(), dataset.outputs[i].end()));

                // Obtener predicción (neurona con mayor activación)
                float* pred_start = h_predictions + i * 10;
                int pred_label = distance(pred_start, max_element(pred_start, pred_start + 10));

                confusion_matrix[true_label][pred_label]++;
            }

            // 6. Guardar matriz en archivo
            ofstream matrix_file("confusion_matrix.csv");
            matrix_file << "Real\\Pred,0,1,2,3,4,5,6,7,8,9\n";
            for (int i = 0; i < 10; ++i) {
                matrix_file << i;
                for (int j = 0; j < 10; ++j) {
                    matrix_file << "," << confusion_matrix[i][j];
                }
                matrix_file << "\n";
            }
            matrix_file.close();

            // 7. Liberar memoria
            delete[] h_predictions;
            CHECK_CUDA(cudaFree(d_test_input));

            cout << "Matriz de confusión guardada en 'confusion_matrix.csv'" << endl;
        }


    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
