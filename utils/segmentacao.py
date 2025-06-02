import numpy as np

def otsu_binarizacao(imagem):
    hist, _ = np.histogram(imagem.flatten(), bins=256, range=(0, 256))
    total = imagem.size

    soma_total_intensidades = np.dot(np.arange(256), hist)
    soma_fundo = 0
    peso_objeto = 0
    max_var = 0
    limiar_otimo = 0

    for t in range(256):
        peso_objeto += hist[t]
        if peso_objeto == 0:
            continue

        peso_f = total - peso_objeto
        if peso_f == 0:
            break

        soma_fundo += t * hist[t]
        media_fundo = soma_fundo / peso_objeto
        media_objeto = (soma_total_intensidades - soma_fundo) / peso_f

        variancia_entre_classes = peso_objeto * peso_f * (media_fundo - media_objeto) ** 2

        if variancia_entre_classes > max_var:
            max_var = variancia_entre_classes
            limiar_otimo = t

    binaria = np.where(imagem > limiar_otimo, 255, 0).astype(np.uint8)
    return binaria

def crescimento_regiao(imagem_binaria):
    rotulada = np.zeros_like(imagem_binaria, dtype=np.int32)
    rotulo = 1
    altura, largura = imagem_binaria.shape

    for i in range(altura):
        for j in range(largura):
            if imagem_binaria[i, j] == 255 and rotulada[i, j] == 0:
                fila = [(i, j)]
                rotulada[i, j] = rotulo

                while fila:
                    x, y = fila.pop(0)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < altura and 0 <= ny < largura:
                                if imagem_binaria[nx, ny] == 255 and rotulada[nx, ny] == 0:
                                    rotulada[nx, ny] = rotulo
                                    fila.append((nx, ny))
                rotulo += 1

    return rotulada, rotulo - 1