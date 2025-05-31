import numpy as np
import cv2

def erosao(imagem, kernel=np.ones((3, 3), dtype=np.uint8)):
    pad = kernel.shape[0] // 2
    imagem_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    resultado = np.zeros_like(imagem)

    for i in range(pad, imagem.shape[0] + pad):
        for j in range(pad, imagem.shape[1] + pad):
            regiao = imagem_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            if np.all(regiao[kernel == 1] == 255):
                resultado[i - pad, j - pad] = 255
    return resultado

def dilatacao(imagem, kernel=np.ones((3, 3), dtype=np.uint8)):
    pad = kernel.shape[0] // 2
    imagem_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    resultado = np.zeros_like(imagem)

    for i in range(pad, imagem.shape[0] + pad):
        for j in range(pad, imagem.shape[1] + pad):
            regiao = imagem_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            if np.any(regiao[kernel == 1] == 255):
                resultado[i - pad, j - pad] = 255
    return resultado

def abertura(imagem, kernel=np.ones((3, 3), dtype=np.uint8)):
    return dilatacao(erosao(imagem, kernel), kernel)

def fechamento(imagem, kernel=np.ones((3, 3), dtype=np.uint8)):
    return erosao(dilatacao(imagem, kernel), kernel)