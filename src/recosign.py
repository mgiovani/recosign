import cv2 as cv
imagem = cv.imread('../data/assinaturas.png')
cv.imshow("Folha de assinaturas", imagem)
cv.waitKey(0)