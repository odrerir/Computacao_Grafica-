import numpy as np
import cv2

def media(imagem, tamanho=3):
    pad = tamanho // 2
    imagem_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    resultado = np.zeros_like(imagem)

    for i in range(pad, imagem.shape[0] + pad):
        for j in range(pad, imagem.shape[1] + pad):
            regiao = imagem_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            media = np.mean(regiao)
            resultado[i - pad, j - pad] = media
    return resultado.astype(np.uint8)

def mediana(imagem, tamanho=3):
    pad = tamanho // 2
    imagem_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    resultado = np.zeros_like(imagem)

    for i in range(pad, imagem.shape[0] + pad):
        for j in range(pad, imagem.shape[1] + pad):
            regiao = imagem_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            mediana = np.median(regiao)
            resultado[i - pad, j - pad] = mediana
    return resultado.astype(np.uint8)

def canny(imagem, limiar=100):
    # Passo 1: Suavização com filtro da média
    suavizada = media(imagem)
    suavizada = suavizada.astype(np.int32)

    # Passo 2: Cálculo de gradientes com Sobel (manualmente)
    altura, largura = suavizada.shape
    grad_x = np.zeros_like(suavizada, dtype=np.float32)
    grad_y = np.zeros_like(suavizada, dtype=np.float32)

    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]

    sobel_y = [[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]]

    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            gx = 0
            gy = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    valor = suavizada[i + m, j + n]
                    gx += valor * sobel_x[m + 1][n + 1]
                    gy += valor * sobel_y[m + 1][n + 1]
            grad_x[i, j] = gx
            grad_y[i, j] = gy

    # Passo 3: Magnitude do gradiente
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Passo 4: Limiarização (binarização)
    bordas = np.where(magnitude >= limiar, 255, 0).astype(np.uint8)

    return bordas
