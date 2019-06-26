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
cv.imshow("Imagem redimensionada", imagem_redimensionada)

# Recorte
assinaturas = []
altura_retangulo = 85
assinaturas.append(imagem_redimensionada[76:158, 76:623])
for i in range(160, 925, altura_retangulo):
    assinaturas.append(imagem_redimensionada[i:i+altura_retangulo, 76:615])

for assinatura in assinaturas:
    # Binarização da assinatura
    for i in range(assinatura.shape[0]):
        for j in range(assinatura.shape[1]):
            if assinatura[i][j] < 220:
                assinatura[i][j] = 0
            else:
                assinatura[i][j] = 255

    # Extração de cantos da assinatura
    quinas = cv.cornerHarris(assinatura,2,3,0.04)
    # Caracteristicas
    for i in range(quinas.shape[0]):
        media = np.sum(quinas[i])/quinas.shape[0]

    print(quinas)
    cv.imshow("Imagem quina", quinas)
    cv.waitKey(0)