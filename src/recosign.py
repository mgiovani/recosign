import cv2 as cv

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
assinaturas.append(imagem_redimensionada[74:160, 74:625])
for i in range(160, 925, altura_retangulo):
    assinaturas.append(imagem_redimensionada[i:i+altura_retangulo, 74:625])

for assinatura in assinaturas:
    cv.imshow("Assinatura", assinatura)
    # Extração de cantos da assinatura
    quinas = cv.cornerHarris(assinatura,2,3,0.04)
    print("quinas: ", quinas)
    cv.waitKey(0)