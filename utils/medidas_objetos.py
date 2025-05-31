import numpy as np
import cv2
import matplotlib.pyplot as plt

def calcular_area(rotulada, rotulo):
    return np.sum(rotulada == rotulo)

def calcular_perimetro(rotulada, rotulo):
    objeto = np.uint8(rotulada == rotulo) * 255
    contornos, _ = cv2.findContours(objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        return cv2.arcLength(contornos[0], True)
    return 0

def calcular_diametro(rotulada, rotulo):
    objeto = np.uint8(rotulada == rotulo) * 255
    contornos, _ = cv2.findContours(objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        pontos = contornos[0][:, 0, :]
        max_dist = 0
        for i in range(len(pontos)):
            for j in range(i + 1, len(pontos)):
                dist = np.linalg.norm(pontos[i] - pontos[j])
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    return 0

def medidas_por_objeto(rotulada):
    rotulos = np.unique(rotulada)
    rotulos = rotulos[rotulos != 0]
    medidas = []
    for r in rotulos:
        area = calcular_area(rotulada, r)
        perimetro = calcular_perimetro(rotulada, r)
        diametro = calcular_diametro(rotulada, r)
        medidas.append({
            'rotulo': r,
            'area': area,
            'perimetro': perimetro,
            'diametro': diametro
        })
    return medidas

def plot_histogram(hist, title, filename):
    plt.figure()
    plt.bar(range(len(hist)), hist)
    plt.title(title)
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.savefig(filename)
    plt.show()

def histograma(img):
  h = np.zeros(256, dtype=int)
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          h[img[i, j]] += 1
  return h