# -*- coding: utf-8 -*-
# shape 0 linha
# shape 1 coluna
import cv2 as cv
import numpy as np

# Visualizar a matriz completa no print
np.set_printoptions(threshold=np.inf)

imagem = cv.imread('../data/assinaturas.jpg', 0)

# Filtro para redução de ruídos
imagem = cv.medianBlur(imagem, 3)
imagem = cv.GaussianBlur(imagem, (9, 9), 0)

# Redimensionamento
proporcao = 700.0 / imagem.shape[1]
tamanho_novo = (700, int(imagem.shape[0] * proporcao))
imagem_redimensionada = cv.resize(imagem, tamanho_novo, interpolation = cv.INTER_AREA)
# cv.imshow("Imagem redimensionada", imagem_redimensionada)

# Recorte
assinaturas = []
altura_retangulo = 85
assinaturas.append(imagem_redimensionada[76:158, 76:623])
for i in range(160, 925, altura_retangulo):
    assinaturas.append(imagem_redimensionada[i+5:i+altura_retangulo-5, 76:615])

for assinatura in assinaturas:
    # cv.imshow("Assinatura cortada", assinatura)

    # Binarização da assinatura
    for i in range(assinatura.shape[0]):
        for j in range(assinatura.shape[1]):
            if assinatura[i][j] < 220:
                assinatura[i][j] = 0
            else:
                assinatura[i][j] = 255
    
    #Fit(recorte) da assinatura nas laterais
    x1 = None
    for j in range(assinatura.shape[1]):
        for i in range(assinatura.shape[0]):
            if (assinatura[i][j] != 255):
                x1 = j
                break
        if x1 != None:
            break
    x2 = None
    for j in range(assinatura.shape[1]-1, 0, -1):
        for i in range(assinatura.shape[0]-1, 0, -1):
            if (assinatura[i][j] != 255):
                x2 = j
                break
        if x2 != None:
            break

    assinatura = assinatura[0:assinatura.shape[0], x1:x2]
    
    #Extrai caracteristicas
    caracteristicas = [0] * assinatura.shape[1]
    for j in range(assinatura.shape[1]):
        for i in range(assinatura.shape[0]):
            if assinatura[i][j] == 0:
                caracteristicas[j] += 1

    #Trata os vetores de caracteristicas fixando a dimensao em 300
    if len(caracteristicas) < 300:
        for _ in range(len(caracteristicas)-1, 300):
            caracteristicas.append(0)
    else:
        caracteristicas = caracteristicas[0:300]

    print(caracteristicas)

    cv.imshow("Imagem quina", assinatura)
    cv.waitKey(0)
