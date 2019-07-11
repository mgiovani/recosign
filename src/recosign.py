# -*- coding: utf-8 -*-
# shape 0 linha
# shape 1 coluna
import cv2 as cv
import numpy as np

# Visualizar a matriz completa no print
np.set_printoptions(threshold=np.inf)

for pasta in range(32):

    for imagem in range(5):
        nome_arquivo = '../data/imagens/treino/' + str(pasta+1) + '/0' + str(imagem+1) + '.PNG'
        # nome_arquivo = '../data/imagens/teste/' + str(pasta+1) + '.PNG'
        print(nome_arquivo)
        assinatura = cv.imread(nome_arquivo, 0)

        # Filtro para redução de ruídos
        assinatura = cv.medianBlur(assinatura, 3)
        assinatura = cv.GaussianBlur(assinatura, (5, 5), 0)

        # Extração de bordas
        # assinatura = cv.Canny(assinatura, 50, 100)

        # Binarização da assinatura
        for i in range(assinatura.shape[0]):
            for j in range(assinatura.shape[1]):
                if assinatura[i][j] < 10:
                    assinatura[i][j] = 255
                else:
                    assinatura[i][j] = 0

                # if assinatura[i][j] < 230:
                #     assinatura[i][j] = 0
                # else:
                #     assinatura[i][j] = 255

        # Fit(recorte) da assinatura nas laterais
        x1 = None
        for j in range(assinatura.shape[1]):
            for i in range(assinatura.shape[0]):
                if (assinatura[i][j] != 255):
                    x1 = j
                    break
            if x1 is not None:
                break
        x2 = None
        for j in range(assinatura.shape[1]-1, 0, -1):
            for i in range(assinatura.shape[0]-1, 0, -1):
                if (assinatura[i][j] != 255):
                    x2 = j
                    break
            if x2 is not None:
                break

        assinatura = assinatura[0:assinatura.shape[0], x1:x2]

        # Extrai caracteristicas
        caracteristicas = [0] * assinatura.shape[1]
        for j in range(assinatura.shape[1]):
            for i in range(assinatura.shape[0]):
                if assinatura[i][j] == 0:
                    caracteristicas[j] += 1

        # Trata os vetores de caracteristicas fixando a dimensao em 300
        aux = []
        for i in range(300):
            if i < len(caracteristicas):
                aux.append(caracteristicas[i])
            else:
                aux.append(0)

        caracteristicas = aux

        arq_vetor = open('../data/vetores/treino/vetores.txt', 'a')
        # arq_classes = open('../data/vetores/treino/classes.txt', 'a')
        arq_vetor.write(str(caracteristicas))
        # arq_classes.write(str(pasta))
        arq_vetor.write(',\n')
        # arq_classes.write('\n')

        cv.waitKey(0)
        arq_vetor.close()
        # arq_classes.close()
