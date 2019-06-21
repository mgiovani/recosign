import cv2 as cv

imagem = cv.imread('../data/assinaturas.jpg')

# Redimensionamento
proporcao = 700.0 / imagem.shape[1]
tamanho_novo = (700, int(imagem.shape[0] * proporcao))
imagem_redimensionada = cv.resize(imagem, tamanho_novo, interpolation = cv.INTER_AREA)
cv.imshow("Imagem redimensionada", imagem_redimensionada)

# Recorte
recorte = imagem_redimensionada[74:160, 74:625]
cv.imshow("Recorte da imagem", recorte)

cv.waitKey(0)