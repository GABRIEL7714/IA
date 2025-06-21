# MLP Acelerado con CUDA (Red Neuronal Multicapa en GPU)

Este proyecto implementa un **perceptrón multicapa (MLP)** en C++ utilizando **CUDA** para acelerar el entrenamiento en **GPU**. Fue diseñado para clasificar imágenes del dataset **MNIST**.

---

## ¿Qué se paraleliza en la GPU?

Durante el entrenamiento, se ejecutan en la GPU las siguientes operaciones:

1. **Propagación hacia adelante (Forward Propagation)**  
   Kernel: `forward_kernel`  
   Calcula las salidas de cada neurona en cada capa.

2. **Cálculo del error en la capa de salida**  
   Kernel: `calcular_error`  
   Calcula la diferencia entre la predicción y la etiqueta deseada.

3. **Retropropagación del error en capas ocultas**  
   Kernel: `calcular_error_ocultos`  
   Propaga el error hacia atrás.

4. **Actualización de pesos y biases**  
   Kernel: `actualizar_pesos`  
   Aplica gradiente descendente y actualiza los pesos con `atomicAdd`.

---

## Configuración de Hilos CUDA

### 1. `forward_kernel`

```cpp
dim3 block(16, 16);
dim3 grid(
    (batch_size + block.x - 1) / block.x,
    (out_size + block.y - 1) / block.y
);
```

- Cada hilo procesa una neurona de salida para una muestra.
- **Paralelismo:** `batch_size × out_size` hilos por capa.

---

### 2. `calcular_error`

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (total_elementos + threadsPerBlock - 1) / threadsPerBlock;
```

- Cada hilo calcula un valor de error de salida.
- **Paralelismo:** `batch_size × output_size` hilos.

---

### 3. `calcular_error_ocultos`

```cpp
dim3 block(16, 16);
dim3 grid(
    (batch_size + block.x - 1) / block.x,
    (current_size + block.y - 1) / block.y
);
```

- Cada hilo calcula el error de una neurona oculta.
- **Paralelismo:** `batch_size × capa_oculta_size` hilos.

---

### 4. `actualizar_pesos`

```cpp
dim3 block(16, 16);
dim3 grid(
    (out_size + block.x - 1) / block.x,
    (in_size + block.y - 1) / block.y
);
```

- Cada hilo actualiza un peso `W_ij` y los biases.
- Se utiliza `atomicAdd` para evitar condiciones de carrera.

---
