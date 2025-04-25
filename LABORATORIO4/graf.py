import matplotlib.pyplot as plt

generaciones = []
mejores = []
promedios = []

with open("resultados.txt", "r") as f:
    for linea in f:
        gen, mejor, prom = map(float, linea.strip().split())
        generaciones.append(gen)
        mejores.append(mejor)
        promedios.append(prom)

plt.plot(generaciones, mejores, label="Mejor distancia", color="green", marker='o')
plt.plot(generaciones, promedios, label="Distancia promedio", color="blue", linestyle='--', marker='x')
plt.xlabel("Generación")
plt.ylabel("Distancia")
plt.title("Distancia por generación - AG TSP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

