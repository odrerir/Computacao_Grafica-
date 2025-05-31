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
    # Passo 1: filtro Gaussiano simples (suavização leve)
    suavizada = media(imagem, tamanho=3)

    # Passo 2: gradientes com Sobel
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

    sobely = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])

    grad_x = cv2.filter2D(suavizada, -1, sobelx)
    grad_y = cv2.filter2D(suavizada, -1, sobely)

    # Magnitude do gradiente
    magnitude = np.sqrt(grad_x.astype(np.float32)**2 + grad_y.astype(np.float32)**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    # Limiarização simples para realçar bordas fortes
    _, binarizada = cv2.threshold(magnitude, limiar, 255, cv2.THRESH_BINARY)
    return binarizada