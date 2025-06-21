#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>
#include <unordered_map>


using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "Error CUDA en " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}

struct Dataset {
    vector<vector<float>> inputs;
    vector<vector<float>> outputs;
};

Dataset cargarCSV(const string& filename, size_t max_datos) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        throw runtime_error("No se puede abrir: " + filename);
    }

    size_t count = 0;
    while (getline(file, line) && count < max_datos) {
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
    cout << "Cargados " << data.inputs.size() << " datos." << endl;
    return data;
}

//CUDA

__global__ void forward_kernel(
    const float* input, const float* pesos, const float* bias,
    float* output, int in_size, int out_size, int batch_size, bool use_relu
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx >= batch_size || neuron_idx >= out_size) return;

    float sum = 0.0f;
    for (int k = 0; k < in_size; ++k) {
        sum += input[sample_idx * in_size + k] * pesos[neuron_idx * in_size + k];
    }
    sum += bias[neuron_idx];

    if (use_relu) {
        sum = fmaxf(0.0f, sum);
    }

    output[sample_idx * out_size + neuron_idx] = sum;
}

__global__ void calcular_error(
    const float* output,
    const float* target,
    float* error,
    int total_elementos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elementos) return;

    error[idx] = output[idx] - target[idx];
}

__global__ void calcular_error_ocultos(
    const float* activation,
    const float* current_error,
    const float* current_pesos,
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
            sum += current_pesos[weight_idx] * current_error[error_idx];
        }
    }

    int output_idx = sample_idx * current_size + neuron_idx;
    if (output_idx < batch_size * current_size) {
        float derivative = (activation[output_idx] > 0.0f) ? 1.0f : 0.0f;
        prev_error[output_idx] = sum * derivative;
    }
}

__global__ void actualizar_pesos(
    float* pesos,
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
        atomicAdd(&pesos[neuron_out * in_size + neuron_in], -lr * grad / batch_size);
    }

    if (neuron_out < out_size && neuron_in == 0) {
        float grad_b = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad_b += error[i * out_size + neuron_out];
        }
        atomicAdd(&bias[neuron_out], -lr * grad_b / batch_size);
    }
}



struct MLP {
    int num_capas;
    vector<int> capas_sizes;
    float taza_aprendizaje;
    int batch_size;

    float** d_pesos;
    float** d_bias;
    float** d_outputs;
    float** d_errors;

    void init(const vector<int>& sizes, float lr, int batch) {
        if (sizes.size() < 2 || lr <= 0 || batch <= 0) {
            throw invalid_argument("MLP invalido");
        }

        num_capas = sizes.size();
        capas_sizes = sizes;
        taza_aprendizaje = lr;
        batch_size = batch;

        d_pesos = new float* [num_capas - 1]();
        d_bias = new float* [num_capas - 1]();
        d_outputs = new float* [num_capas]();
        d_errors = new float* [num_capas - 1]();

        for (int i = 0; i < num_capas; ++i) {
            CHECK_CUDA(cudaMalloc(&d_outputs[i], sizeof(float) * capas_sizes[i] * batch_size));

            cout << "Asignado d_outputs[" << i << "] con size " << capas_sizes[i] * batch_size << endl;
        }

        for (int i = 0; i < num_capas - 1; ++i) {
            int in_size = capas_sizes[i];
            int out_size = capas_sizes[i + 1];

            CHECK_CUDA(cudaMalloc(&d_pesos[i], sizeof(float) * in_size * out_size));
            CHECK_CUDA(cudaMalloc(&d_bias[i], sizeof(float) * out_size));
            CHECK_CUDA(cudaMalloc(&d_errors[i], sizeof(float) * out_size * batch_size));

            cout << "Capas asignadas " << i << " pesos: " << in_size << "x" << out_size << ", errores: " << out_size * batch_size << endl;

            float* h_pesos = new float[in_size * out_size];
            for (int j = 0; j < in_size * out_size; ++j) {
                h_pesos[j] = (rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            }
            CHECK_CUDA(cudaMemcpy(d_pesos[i], h_pesos, sizeof(float) * in_size * out_size, cudaMemcpyHostToDevice));
            delete[] h_pesos;

            float* h_bias = new float[out_size];
            for (int j = 0; j < out_size; ++j) {
                h_bias[j] = (rand() / (float)RAND_MAX) * 0.2f - 0.1f;
            }
            CHECK_CUDA(cudaMemcpy(d_bias[i], h_bias, sizeof(float) * out_size, cudaMemcpyHostToDevice));
            delete[] h_bias;
        }
    }

    void free() {
        for (int i = 0; i < num_capas - 1; ++i) {
            if (d_pesos[i]) cudaFree(d_pesos[i]);
            if (d_bias[i]) cudaFree(d_bias[i]);
            if (d_errors[i]) cudaFree(d_errors[i]);
        }
        for (int i = 0; i < num_capas; ++i) {
            if (d_outputs[i]) cudaFree(d_outputs[i]);
        }

        delete[] d_pesos;
        delete[] d_bias;
        delete[] d_outputs;
        delete[] d_errors;
    }

    void forward(float* d_input, int current_batch_size) {
        if (current_batch_size <= 0) {
            throw invalid_argument("batch size invalid en el forward()");
        }

        size_t input_bytes = sizeof(float) * current_batch_size * capas_sizes[0];

        CHECK_CUDA(cudaMemcpy(
            d_outputs[0],          
            d_input,              
            input_bytes,           
            cudaMemcpyDeviceToDevice  
        ));
        for (int l = 0; l < num_capas - 1; ++l) {
            int in_size = capas_sizes[l];
            int out_size = capas_sizes[l + 1];
            bool is_output_layer = (l == num_capas - 2);

            dim3 block(16, 16);
            dim3 grid(
                (current_batch_size + block.x - 1) / block.x,
                (out_size + block.y - 1) / block.y
            );

            cout << "Forward capa " << l << " grid: (" << grid.x << ", " << grid.y << ")" << endl;

            forward_kernel << <grid, block >> > (
                d_outputs[l],
                d_pesos[l],
                d_bias[l],
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
        if (current_batch_size <= 0 || current_batch_size > batch_size) {
            throw invalid_argument("batch size invalido en backward()");
        }

        int output_layer_idx = num_capas - 1;
        int output_size = capas_sizes[output_layer_idx];
        int total_output = current_batch_size * output_size;

        if (output_layer_idx - 1 < 0 || output_layer_idx - 1 >= num_capas - 1) {
            throw runtime_error("capa invalidaen backward()");
        }


        int threadsPerBlock = 256;
        int blocksPerGrid = (total_output + threadsPerBlock - 1) / threadsPerBlock;

        calcular_error << <blocksPerGrid, threadsPerBlock >> > (
            d_outputs[output_layer_idx],
            d_target,
            d_errors[output_layer_idx - 1],
            total_output
            );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        for (int l = num_capas - 2; l >= 1; --l) {
            int current_size = capas_sizes[l];
            int next_size = capas_sizes[l + 1];

            dim3 block(16, 16);
            dim3 grid(
                (current_batch_size + block.x - 1) / block.x,
                (current_size + block.y - 1) / block.y
            );


            calcular_error_ocultos << <grid, block >> > (
                d_outputs[l],
                d_errors[l],
                d_pesos[l],
                d_errors[l - 1],
                current_size,
                next_size,
                current_batch_size
                );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        for (int l = 0; l < num_capas - 1; ++l) {
            int in_size = capas_sizes[l];
            int out_size = capas_sizes[l + 1];

            dim3 block(16, 16);
            dim3 grid(
                (out_size + block.x - 1) / block.x,
                (in_size + block.y - 1) / block.y
            );

            const float* input = (l == 0) ? d_input : d_outputs[l];

            actualizar_pesos << <grid, block >> > (
                d_pesos[l],
                d_bias[l],
                input,
                d_errors[l],
                taza_aprendizaje,
                in_size,
                out_size,
                current_batch_size
                );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    float compute_error(float* d_target, int current_batch_size) {
        int output_size = capas_sizes.back();
        int total = current_batch_size * output_size;

        float* d_error;
        CHECK_CUDA(cudaMalloc(&d_error, sizeof(float) * total));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        calcular_error << <blocks, threads >> > (
            d_outputs[num_capas - 1],
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

void test_and_confusion_matrix(MLP& net, const string& test_filename, int test_samples) {
    Dataset test_data = cargarCSV(test_filename, test_samples);

    float* h_test_inputs = new float[test_samples * 784];
    float* h_test_outputs = new float[test_samples * 10];

    for (int i = 0; i < test_samples; ++i) {
        copy(test_data.inputs[i].begin(), test_data.inputs[i].end(), h_test_inputs + i * 784);
        copy(test_data.outputs[i].begin(), test_data.outputs[i].end(), h_test_outputs + i * 10);
    }

    float* d_test_inputs, * d_test_outputs;
    CHECK_CUDA(cudaMalloc(&d_test_inputs, sizeof(float) * test_samples * 784));
    CHECK_CUDA(cudaMalloc(&d_test_outputs, sizeof(float) * test_samples * 10));
    CHECK_CUDA(cudaMemcpy(d_test_inputs, h_test_inputs, sizeof(float) * test_samples * 784, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_test_outputs, h_test_outputs, sizeof(float) * test_samples * 10, cudaMemcpyHostToDevice));

    net.forward(d_test_inputs, test_samples);

    float* h_predicted = new float[test_samples * 10];
    CHECK_CUDA(cudaMemcpy(h_predicted, net.d_outputs[net.num_capas - 1],
        sizeof(float) * test_samples * 10, cudaMemcpyDeviceToHost));

    int confusion[10][10] = { 0 };
    int correct = 0;

    for (int i = 0; i < test_samples; ++i) {
        int true_label = 0;
        for (int j = 0; j < 10; ++j) {
            if (test_data.outputs[i][j] == 1.0f) {
                true_label = j;
                break;
            }
        }

        int predicted_label = 0;
        float max_val = h_predicted[i * 10];
        for (int j = 1; j < 10; ++j) {
            if (h_predicted[i * 10 + j] > max_val) {
                max_val = h_predicted[i * 10 + j];
                predicted_label = j;
            }
        }

        confusion[true_label][predicted_label]++;

        if (true_label == predicted_label) {
            correct++;
        }
    }

    float accuracy = static_cast<float>(correct) / test_samples * 100.0f;

    cout << "\nMatriz de Confusion:\n";
    cout << "Real \\ Predic  0     1     2     3     4     5     6     7     8     9\n";
    for (int i = 0; i < 10; ++i) {
        cout << i << "        ";
        for (int j = 0; j < 10; ++j) {
            cout << confusion[i][j];
            if (confusion[i][j] < 10) cout << "    ";
            else if (confusion[i][j] < 100) cout << "   ";
            else cout << "  ";
        }
        cout << "\n";
    }

    cout << "\nPrecision general: " << accuracy << "%\n";

    cout << "\nMetricas por clase:\n";
    cout << "Clase\tPrecision\tRecall\tF1-Score\n";
    for (int i = 0; i < 10; ++i) {
        int true_positives = confusion[i][i];
        int false_positives = 0;
        int false_negatives = 0;

        for (int j = 0; j < 10; ++j) {
            if (j != i) {
                false_positives += confusion[j][i];
                false_negatives += confusion[i][j];
            }
        }

        float precision = (true_positives + false_positives) == 0 ? 0 :
            static_cast<float>(true_positives) / (true_positives + false_positives);
        float recall = (true_positives + false_negatives) == 0 ? 0 :
            static_cast<float>(true_positives) / (true_positives + false_negatives);
        float f1 = (precision + recall) == 0 ? 0 :
            2 * (precision * recall) / (precision + recall);

        cout << i << "\t" << precision * 100 << "%\t\t"
            << recall * 100 << "%\t" << f1 * 100 << "%\n";
    }

    ofstream conf_file("confusion_matrix.csv");
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            conf_file << confusion[i][j];
            if (j < 9) conf_file << ",";
        }
        conf_file << "\n";
    }
    conf_file.close();

    delete[] h_test_inputs;
    delete[] h_test_outputs;
    delete[] h_predicted;
    CHECK_CUDA(cudaFree(d_test_inputs));
    CHECK_CUDA(cudaFree(d_test_outputs));
}


int main() {
    try {
        string train_csv = "mnist.csv";
        string test_csv = "mnist_test.csv";
        int total_samples = 60000;
        int test_samples = 10000;
        int batch_size = 1000;
        float taza_aprendizaje = 0.01f;
        int epochs = 500;
        vector<int> architecture = { 784, 128, 10 };

        cout << "Cargando dataset de entrenamiento..." << endl;
        Dataset dataset = cargarCSV(train_csv, total_samples);


        int num_batches = total_samples / batch_size;

        cout << "Preparando GPU data..." << endl;
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

        cout << "Iniciando MLP..." << endl;
        MLP net;
        net.init(architecture, taza_aprendizaje, batch_size);

        cout << "Entrenando..." << endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_error = 0.0f;

            for (int i = 0; i < num_batches; ++i) {
                int input_offset = i * batch_size * 784; 
                int output_offset = i * batch_size * 10; 

                float* d_batch_in = d_inputs + input_offset;
                float* d_batch_out = d_outputs + output_offset;

                cout << "Epoca " << epoch << ", Batch " << i << endl;

                net.forward(d_batch_in, batch_size);
                net.backward(d_batch_in, d_batch_out, batch_size);

                float batch_error = net.compute_error(d_batch_out, batch_size);
                total_error += batch_error;

                cout << "Batch error: " << batch_error << endl;
                float prec = 1.0 - batch_error;
                prec = prec * 100;
                cout << "Precision: " << prec << "%\n";
            }

            float avg_error = total_error / num_batches;
            net.log_error(epoch, avg_error, "error.csv");
            cout << "Epoca " << epoch << " - Error: " << avg_error << endl;
        }

        cout << "Limpiando memoria..." << endl;
        net.free();
        CHECK_CUDA(cudaFree(d_inputs));
        CHECK_CUDA(cudaFree(d_outputs));

        cout << "Entrenamiento completo!" << endl;
        net.init(architecture, taza_aprendizaje, test_samples);

        cout << "\nProbando con datos de test..." << endl;
        test_and_confusion_matrix(net, test_csv, test_samples);

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
