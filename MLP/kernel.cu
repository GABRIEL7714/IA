#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>


using namespace std;

//CUDA

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error en " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

//Leer datos CSV

struct leerData {
    vector<vector<float>> inputs;
    vector<vector<float>> outputs;

};

leerData cargarCsv(const string& archivo_csv, size_t max_ejemplos) {
    leerData dataset;
    ifstream archivo(archivo_csv);
    string linea;

    if (!archivo.is_open()) {
        throw runtime_error("No se pudo abrir el archivo CSV: " + archivo_csv);
    }

    size_t contador = 0;
    while (getline(archivo, linea) && contador < max_ejemplos) {
        stringstream ss(linea);
        string valor;

        vector<float> input(784);
        vector<float> output(10, 0.0f);

        // Leer etiqueta (valor deseado)
        getline(ss, valor, ',');
        int etiqueta = stoi(valor);
        if (etiqueta < 0 || etiqueta > 9) continue;
        output[etiqueta] = 1.0f;

        // Leer 784 pixeles
        for (int i = 0; i < 784; ++i) {
            if (!getline(ss, valor, ',')) break;
            input[i] = stof(valor) / 255.0f;  // normalizar [0,1]
        }

        dataset.inputs.push_back(move(input));
        dataset.outputs.push_back(move(output));
        ++contador;
    }

    cout << "Cargadas " << dataset.inputs.size() << " imagenes desde " << archivo_csv << "\n";
    return dataset;
}

//MLP

struct MLP {
    int num_capas;                          // Ej: 4 (784, 256, 128, 10)
    vector<int> capas_sizes;            // Tamaño de cada capa
    float tazaAprendizaje;                     // Tasa de aprendizaje

    // Pesos y bias (en GPU)
    float** d_pesos;  // d_weights[i] es matriz [layer_sizes[i+1] x layer_sizes[i]]
    float** d_bias;   // d_biases[i] es vector [layer_sizes[i+1]]

    // Salidas temporales (activaciones por capa)
    float** d_outputs;  // d_outputs[i] es vector [layer_sizes[i]]

    // Errores (para backprop)
    float** d_errores;

    // Inicializa la red y reserva memoria en GPU
    void inicializar(const vector<int>& sizes, float lr);

    // Libera toda la memoria
    void liberar();

    // Forward propagation (toda la batch)
    void forward(float* d_input_batch, int batch_size);

    // Backward propagation y actualización
    void backward(float* d_input_batch, float* d_target_batch, int batch_size);

    // Calcula el error cuadrático total
    float calcular_error(float* d_target_batch, int batch_size);

    // Guarda error en archivo .txt
    void log_error(int epoca, float error, const string& nombre_archivo);
};

void MLP::inicializar(const std::vector<int>& sizes, float lr) {
    num_capas = sizes.size();
    capas_sizes = sizes;
    tazaAprendizaje = lr;

    // Reservar punteros para cada capa (pesos, bias, activaciones, deltas)
    d_pesos = new float* [num_capas - 1];
    d_bias = new float* [num_capas - 1];
    d_outputs = new float* [num_capas];
    d_errores = new float* [num_capas - 1];

    // Reservar espacio para salidas de cada capa
    for (int i = 0; i < num_capas; ++i) {
        CHECK_CUDA(cudaMalloc(&d_outputs[i], sizeof(float) * capas_sizes[i] * 1000));
    }

    // Inicializar pesos y bias aleatoriamente [-0.1, 0.1]
    for (int i = 0; i < num_capas - 1; ++i) {
        int in_size = capas_sizes[i];
        int out_size = capas_sizes[i + 1];

        CHECK_CUDA(cudaMalloc(&d_pesos[i], sizeof(float) * in_size * out_size));
        CHECK_CUDA(cudaMalloc(&d_bias[i], sizeof(float) * out_size));
        CHECK_CUDA(cudaMalloc(&d_errores[i], sizeof(float) * out_size * 1000));

        // Inicializar aleatorio en GPU
        float* h_temp = new float[in_size * out_size];
        for (int j = 0; j < in_size * out_size; ++j) {
            h_temp[j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        }
        CHECK_CUDA(cudaMemcpy(d_pesos[i], h_temp, sizeof(float) * in_size * out_size, cudaMemcpyHostToDevice));
        delete[] h_temp;

        float* h_bias = new float[out_size];
        for (int j = 0; j < out_size; ++j) {
            h_bias[j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        }
        CHECK_CUDA(cudaMemcpy(d_bias[i], h_bias, sizeof(float) * out_size, cudaMemcpyHostToDevice));
        delete[] h_bias;
    }
}

void MLP::liberar() {
    for (int i = 0; i < num_capas - 1; ++i) {
        cudaFree(d_pesos[i]);
        cudaFree(d_bias[i]);
        cudaFree(d_errores[i]);
    }
    for (int i = 0; i < num_capas; ++i) {
        cudaFree(d_outputs[i]);
    }
    delete[] d_pesos;
    delete[] d_bias;
    delete[] d_errores;
    delete[] d_outputs;
}

void MLP::log_error(int epoca, float error, const string& nombre_archivo) {
    ofstream archivo(nombre_archivo, ios::app);
    archivo << "Epoca " << epoca << ": " << error << "\n";
    archivo.close();
}

//CUDA

__global__ void forward_layer_kernel(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output, int input_size, int output_size, int batch_size, bool aplicar_relu
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // índice de imagen (batch)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // neurona de salida

    if (i < batch_size && j < output_size) {
        float sum = 0.0f;
        for (int k = 0; k < input_size; ++k) {
            float x = d_input[i * input_size + k];
            float w = d_weights[j * input_size + k];
            sum += x * w;
        }
        sum += d_bias[j];

        // ReLU si no es capa de salida
        if (aplicar_relu)
            sum = fmaxf(0.0f, sum);

        d_output[i * output_size + j] = sum;
    }
}



void MLP::forward(float* d_input_batch, int batch_size) {
    // Copiar entradas a la capa 0
    CHECK_CUDA(cudaMemcpy(d_outputs[0], d_input_batch, sizeof(float) * batch_size * capas_sizes[0], cudaMemcpyDeviceToDevice));

    // Por cada capa L = 1 ... N
    for (int l = 0; l < num_capas - 1; ++l) {
        int in_size = capas_sizes[l];
        int out_size = capas_sizes[l + 1];

        dim3 blockDim(16, 16);
        dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x,
            (out_size + blockDim.y - 1) / blockDim.y);

        bool aplicar_relu = (l != num_capas - 2);  // no usar ReLU en la capa final
        forward_layer_kernel << <gridDim, blockDim >> > (
            d_outputs[l], d_pesos[l], d_bias[l],
            d_outputs[l + 1], in_size, out_size, batch_size, aplicar_relu
            );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

__global__ void calcular_error_kernel(
    const float* d_salidas, const float* d_targets,
    float* d_error_parcial, int total_neuronas)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_neuronas) {
        float diff = d_targets[idx] - d_salidas[idx];
        d_error_parcial[idx] = diff * diff;
    }
}

float MLP::calcular_error(float* d_target_batch, int batch_size) {
    int output_size = capas_sizes.back();
    int total_neuronas = batch_size * output_size;

    float* d_error_parcial;
    CHECK_CUDA(cudaMalloc(&d_error_parcial, sizeof(float) * total_neuronas));

    int threads = 256;
    int blocks = (total_neuronas + threads - 1) / threads;

    calcular_error_kernel << <blocks, threads >> > (
        d_outputs[num_capas - 1], d_target_batch, d_error_parcial, total_neuronas);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copiar a CPU y sumar
    std::vector<float> h_error_parcial(total_neuronas);
    CHECK_CUDA(cudaMemcpy(h_error_parcial.data(), d_error_parcial, sizeof(float) * total_neuronas, cudaMemcpyDeviceToHost));

    float suma_error = 0.0f;
    for (int i = 0; i < total_neuronas; ++i) {
        suma_error += h_error_parcial[i];
    }

    cudaFree(d_error_parcial);
    return suma_error;
}

__global__ void calcular_delta_salida(
    const float* salida, const float* target,
    float* delta, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float error = salida[idx] - target[idx];
        delta[idx] = error;  // derivada identidad
    }
}
__global__ void calcular_delta_oculta(
    const float* activacion, const float* siguiente_delta,
    const float* siguiente_weights,
    float* delta, int input_size, int output_size, int batch_size)
{
    int img_idx = blockIdx.x * blockDim.x + threadIdx.x;      // Índice en el batch (0 a batch_size-1)
    int neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;   // Índice en la capa actual (0 a input_size-1)

    if (img_idx >= batch_size || neuron_idx >= input_size)
        return;

    float suma = 0.0f;
    for (int k = 0; k < output_size; ++k) {
        // Cálculo seguro de w_idx (pesos entre capa l y l+1)
        int w_idx = neuron_idx * output_size + k;  // ¡Cambiado! Ahora es [input_size][output_size]
        int d_idx = img_idx * output_size + k;     // Delta de la capa siguiente

        // Verificación de límites
        if (w_idx >= input_size * output_size || d_idx >= batch_size * output_size) {
            continue;  // Evita accesos inválidos
        }

        suma += siguiente_weights[w_idx] * siguiente_delta[d_idx];
    }

    int act_idx = img_idx * input_size + neuron_idx;
    if (act_idx >= batch_size * input_size) {
        return;  // Evita escritura fuera de límites
    }

    float a = activacion[act_idx];
    float derivada = (a > 0.0f) ? 1.0f : 0.0f;
    delta[act_idx] = suma * derivada;  // Usamos act_idx para delta también
}

__global__ void actualizar_pesos_bias(
    float* weights, float* bias,
    const float* entrada, const float* delta,
    float lr, int in_size, int out_size, int batch_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // neurona de salida
    int k = blockIdx.y * blockDim.y + threadIdx.y;  // neurona de entrada

    if (j < out_size && k < in_size) {
        float grad = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad += delta[i * out_size + j] * entrada[i * in_size + k];
        }
        weights[j * in_size + k] -= lr * grad / batch_size;
    }

    // Solo el hilo que actualiza el bias
    if (j < out_size && k == 0) {
        float grad_b = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            grad_b += delta[i * out_size + j];
        }
        bias[j] -= lr * grad_b / batch_size;
    }
}

void MLP::backward(float* d_input_batch, float* d_target_batch, int batch_size) {
    int last = num_capas - 1;
    int out_size = capas_sizes[last];

    // Capa de salida
    int total_out = batch_size * out_size;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    calcular_delta_salida << <blocks, threads >> > (
        d_outputs[last], d_target_batch, d_errores[last - 1], total_out
        );
    CHECK_CUDA(cudaDeviceSynchronize());

    // Retropropagación desde la capa penúltima oculta hasta la primera
    for (int l = num_capas - 3; l >= 0; --l) {
        int in_size = capas_sizes[l];
        int out_size = capas_sizes[l + 1];

        // ¡Asegúrate de que l+1 no exceda num_capas - 1!
        if (l + 1 >= num_capas - 1) {
            std::cerr << "Error: Acceso inválido a d_errores[" << l + 1 << "]\n";
            continue;
        }

        // Verifica punteros
        if (!d_outputs[l] || !d_errores[l + 1] || !d_pesos[l + 1] || !d_errores[l]) {
            std::cerr << "Error: Puntero no inicializado en capa " << l << "\n";
            continue;
        }

        dim3 blockDim(16, 16);
        dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x,
            (in_size + blockDim.y - 1) / blockDim.y);

        calcular_delta_oculta << <gridDim, blockDim >> > (
            d_outputs[l], d_errores[l + 1], d_pesos[l + 1], d_errores[l],
            in_size, out_size, batch_size
            );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }


    // Actualizar pesos y bias
    for (int l = 0; l < num_capas - 1; ++l) {
        int in_size = capas_sizes[l];
        int out_size = capas_sizes[l + 1];

        const float* entrada = (l == 0) ? d_input_batch : d_outputs[l];

        dim3 blockDim(16, 16);
        dim3 gridDim((out_size + 15) / 16, (in_size + 15) / 16);
        std::cout << "Layer " << l << ": in=" << in_size << ", out=" << out_size << "\n";

        actualizar_pesos_bias << <gridDim, blockDim >> > (
            d_pesos[l], d_bias[l], entrada, d_errores[l],
            tazaAprendizaje, in_size, out_size, batch_size
            );
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}



float* aplanar_2D(const std::vector<std::vector<float>>& matriz) {
    int filas = matriz.size();
    int columnas = matriz[0].size();
    float* plano = new float[filas * columnas];
    for (int i = 0; i < filas; ++i) {
        std::copy(matriz[i].begin(), matriz[i].end(), plano + i * columnas);
    }
    return plano;
}



int main() {
    std::string ruta_csv = "mnist.csv";  // Asegúrate que esté en tu directorio
    leerData dataset = cargarCsv(ruta_csv, 1000);

    int batch_size = dataset.inputs.size();
    int input_size = dataset.inputs[0].size();   // 784
    int output_size = dataset.outputs[0].size(); // 10

    float* h_inputs = aplanar_2D(dataset.inputs);
    float* h_outputs = aplanar_2D(dataset.outputs);

    float* d_inputs;
    float* d_targets;
    cudaMalloc(&d_inputs, sizeof(float) * batch_size * input_size);
    cudaMalloc(&d_targets, sizeof(float) * batch_size * output_size);

    cudaMemcpy(d_inputs, h_inputs, sizeof(float) * batch_size * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_outputs, sizeof(float) * batch_size * output_size, cudaMemcpyHostToDevice);

    delete[] h_inputs;
    delete[] h_outputs;

    // -----------------------------
    // ⚙️ Configuración de la red
    // -----------------------------
    std::vector<int> arquitectura = { 784, 128, 64, 10 };  // puedes probar con 256, 512 también
    float learning_rate = 0.05f;
    int epocas = 1000;
    std::string archivo_log = "errores_mlp.txt";

    MLP red;
    red.inicializar(arquitectura, learning_rate);

    for (int epoca = 1; epoca <= epocas; ++epoca) {
        red.forward(d_inputs, batch_size);
        red.backward(d_inputs, d_targets, batch_size);

        float error = red.calcular_error(d_targets, batch_size);
        red.log_error(epoca, error, archivo_log);

        if (epoca % 50 == 0 || epoca == 1) {
            std::cout << "Época " << epoca << " - Error: " << error << "\n";
        }
    }

    red.liberar();
    cudaFree(d_inputs);
    cudaFree(d_targets);

    std::cout << "Entrenamiento finalizado. Revisa el archivo " << archivo_log << "\n";
    return 0;
}
