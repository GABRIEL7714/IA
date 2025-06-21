import csv
import matplotlib.pyplot as plt

# Leer archivo CSV
x = []
y = []
with open('error.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= 2:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except ValueError:
                continue  # Saltar filas no numéricas (cabeceras, etc.)

# Graficar
plt.plot(x, y, marker='o')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.title('Gráfico desde CSV')
plt.grid(True)
plt.show()
