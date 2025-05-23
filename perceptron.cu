#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define NUM_NEURONAS 10
#define NUM_ENTRADAS 32
#define N 0.5

using namespace std;

__device__ int f(double x) {
    return x > 0 ? 1 : 0;
}

__global__ void calcularYoutKernel(const int* x, const double* pesos, int* salidas, int entradaSize) {
    int j = threadIdx.x; 

    double sum = 0.0;
    for (int i = 0; i < entradaSize; i++) {
        sum += x[i] * pesos[j * entradaSize + i];
    }
    salidas[j] = f(sum);
}

__global__ void actualizarPesosKernel(double* pesos, const int* x, const int* etiquetas, const int* salidas, int entradaSize, int muestraIdx) {
    int j = threadIdx.x;

    int deseado = (etiquetas[muestraIdx] == j) ? 1 : 0;
    if (salidas[j] != deseado) {
        for (int i = 0; i < entradaSize; i++) {
            pesos[j * entradaSize + i] += N * (deseado - salidas[j]) * x[i];
        }
    }
}

void entrenar(const vector<vector<int>>& entradas, const vector<int>& etiquetas) {
    int numMuestras = entradas.size();
    const int entradaSize = NUM_ENTRADAS + 1;
    const size_t inputSize = entradaSize * sizeof(int);
    const size_t weightSize = NUM_NEURONAS * entradaSize * sizeof(double);
    const size_t salidasSize = NUM_NEURONAS * sizeof(int);

    double* h_pesos = new double[NUM_NEURONAS * entradaSize]();
    double* d_pesos;
    int* d_x;
    int* d_etiquetas;
    int* d_salidas;

    cudaMalloc(&d_pesos, weightSize);
    cudaMalloc(&d_x, inputSize);
    cudaMalloc(&d_etiquetas, numMuestras * sizeof(int));
    cudaMalloc(&d_salidas, salidasSize);

    cudaMemcpy(d_pesos, h_pesos, weightSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_etiquetas, etiquetas.data(), numMuestras * sizeof(int), cudaMemcpyHostToDevice);

    int era = 1;
    bool error;

    do {
        cout << "Era " << era++ << ":\n";
        error = false;
        for (int m = 0; m < numMuestras; m++) {
            vector<int> x = entradas[m];
            x.insert(x.begin(), 1); 
            cudaMemcpy(d_x, x.data(), inputSize, cudaMemcpyHostToDevice);

            calcularYoutKernel << <1, NUM_NEURONAS >> > (d_x, d_pesos, d_salidas, entradaSize);
            cudaDeviceSynchronize();

            int h_salidas[NUM_NEURONAS];
            cudaMemcpy(h_salidas, d_salidas, salidasSize, cudaMemcpyDeviceToHost);

            for (int j = 0; j < NUM_NEURONAS; j++) {
                int deseado = (etiquetas[m] == j) ? 1 : 0;
                //cout << "yout = ";
                for (int i = 0; i < entradaSize; i++) {
                    //cout << x[i] << " x " << h_pesos[j * entradaSize + i] << " + ";
                }
                double sum = 0;
                for (int i = 0; i < entradaSize; i++) {
                    sum += x[i] * h_pesos[j * entradaSize + i];
                }
                //cout << " = " << sum << endl;

                if (h_salidas[j] != deseado) error = true;
            }

            actualizarPesosKernel << <1, NUM_NEURONAS >> > (d_pesos, d_x, d_etiquetas, d_salidas, entradaSize, m);
            cudaDeviceSynchronize();
        }
    } while (error );

    cudaMemcpy(h_pesos, d_pesos, weightSize, cudaMemcpyDeviceToHost);

    cout << "\nEntrenamiento completado.\n\n";
    for (int j = 0; j < NUM_NEURONAS; j++) {
        cout << "Pesos de la neurona " << j << ": ";
        for (int i = 0; i < entradaSize; i++)
            cout << h_pesos[j * entradaSize + i] << " | ";
        cout << "\n";
    }

    cout << "\nEvaluacion:\n";
    for (int m = 0; m < numMuestras; m++) {
        vector<int> x = entradas[m];
        x.insert(x.begin(), 1);
        cudaMemcpy(d_x, x.data(), inputSize, cudaMemcpyHostToDevice);

        calcularYoutKernel << <1, NUM_NEURONAS >> > (d_x, d_pesos, d_salidas, entradaSize);
        cudaDeviceSynchronize();

        int h_salidas[NUM_NEURONAS];
        cudaMemcpy(h_salidas, d_salidas, salidasSize, cudaMemcpyDeviceToHost);

        cout << "Entrada " << m << " (esperado " << etiquetas[m] << ") = salida: ";
        for (int j = 0; j < NUM_NEURONAS; j++) {
            vector<int> x = entradas[m];
            x.insert(x.begin(), 1); 

            double sum = 0;
            cout << "\n  Neurona " << j << ": (";
            for (int i = 0; i < entradaSize; i++) {
                sum += x[i] * h_pesos[j * entradaSize + i];
                //cout << x[i] << "*" << h_pesos[j * entradaSize + i];
                //if (i < entradaSize - 1) cout << " + ";
            }
            cout << ") = " << sum << " = f(sum) = " << h_salidas[j];
        }
        cout << "\n";


        cout << "Entrada " << m << " (esperado " << etiquetas[m] << ") = salida: ";
        for (int j = 0; j < NUM_NEURONAS; j++) cout << h_salidas[j] << " ";
        cout << "\n";
    }

    delete[] h_pesos;
    cudaFree(d_pesos);
    cudaFree(d_x);
    cudaFree(d_etiquetas);
    cudaFree(d_salidas);
}

int main() {
    vector<vector<int>> entradas = {
        {1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1},
        {0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,1},
        {1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1},
        {1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1},
        {1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1},
        {1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1},
        {1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1},
        {1,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0},
        {1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1},
        {1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1}
    };
    vector<int> etiquetas = { 0,1,2,3,4,5,6,7,8,9 };

    entrenar(entradas, etiquetas);
    return 0;
}
