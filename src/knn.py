class Knn():
    k = 1
    # Representação do tipo elemento, classe
    elementos: list = []

    def __init__(self, k):
        self.k = k

    def calcula_distancia(self, vetor_1, vetor_2) -> float:
        distancia = 0.0
        for i in range(len(vetor_1)):
            distancia += (vetor_1[i] - vetor_2[i]) ** 2
        return distancia ** (1/2)
    
    def retorna_distancia(self, elemento) -> float:
        return elemento[0]

    def get_vizinhos(self, centro_consulta) -> list:
        vizinhos: list = []
        k_vizinhos: list = []
        for elem in self.elementos:
            distancia = self.calcula_distancia(centro_consulta[0], elem[0])
            vizinhos.append([distancia, elem])
        vizinhos.sort(key=self.retorna_distancia)

        for i in range(self.k):
            k_vizinhos.append(vizinhos[i][1])

        return k_vizinhos

    def votacao(self, vizinhos) -> str:
        classes = {}
        for v in vizinhos:
            if v[1] in classes.keys():
                classes[v[1]] += 1
            else:
                classes[v[1]] = 1
        mais_votado = max(classes.keys(), key=(lambda i: classes[i]))
        return mais_votado

    def treina(self, elementos) -> None:
        self.elementos = elementos

    def prediz(self, centro_consulta) -> str:
        vizinhos = self.get_vizinhos(centro_consulta)
        classe = self.votacao(vizinhos)
        return classe