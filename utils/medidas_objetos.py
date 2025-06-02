import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import filtros

def calcular_area(rotulada, rotulo):
    return np.sum(rotulada == rotulo)

def encontrar_contornos(imagem_bordas):
    altura, largura = imagem_bordas.shape
    visitado = np.zeros_like(imagem_bordas, dtype=bool)
    contornos = []

    for i in range(altura):
        for j in range(largura):
            if imagem_bordas[i, j] == 255 and not visitado[i, j]:
                pilha = [(i, j)]
                contorno = []

                while pilha:
                    y, x = pilha.pop()
                    if 0 <= y < altura and 0 <= x < largura:
                        if imagem_bordas[y, x] == 255 and not visitado[y, x]:
                            visitado[y, x] = True
                            contorno.append((x, y))

                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dx != 0 or dy != 0:
                                        pilha.append((y + dy, x + dx))

                if len(contorno) > 0:
                    contornos.append(contorno)

    return contornos

def calcular_perimetro(rotulada, rotulo):
    objeto = np.uint8(rotulada == rotulo) * 255
    contornos = cv2.findContours(objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        contorno = contornos[0]
        perimetro = 0
        for i in range(len(contorno) - 1):
            x1, y1 = contorno[i]
            x2, y2 = contorno[i + 1]
            perimetro += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return perimetro
    return 0

def calcular_diametro(rotulada, rotulo):
    objeto = np.uint8(rotulada == rotulo) * 255
    contornos = cv2.findContours(objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        pontos = contornos[0]
        max_dist = 0
        for i in range(len(pontos)):
            for j in range(i + 1, len(pontos)):
                dist = np.linalg.norm(np.array(pontos[i]) - np.array(pontos[j]))
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    return 0


def medidas_por_objeto(rotulada):
    rotulos = np.unique(rotulada)
    rotulos = rotulos[rotulos != 0]
    medidas = []
    for r in rotulos:
        #objeto = np.uint8(rotulada == r) * 255
        #bordas = filtros.canny(objeto)
        #contornos = encontrar_contornos(bordas)
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