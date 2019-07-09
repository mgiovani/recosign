from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix  
from data import vetores, classes 
import knn


knn = knn.Knn(5)

for i in range(len(vetores)):
    vetores[i] = vetores[i][0:300]

x_treino, x_teste, y_treino, y_teste = train_test_split(vetores, classes, test_size=0.33, random_state=42)

x_treino = list(x_treino)
x_teste = list(x_teste)

dados = []
for i in range(len(x_treino)):
    dados.append([list(x_treino[i]), y_treino[i]])

knn.treina(dados)

y_predito = []
for i in range(len(x_teste)):
    y_predito.append(knn.prediz([x_teste[i], y_teste[i]]))

print(y_predito)
print(y_teste)


print(confusion_matrix(y_teste,y_predito))  
print(classification_report(y_teste,y_predito))
