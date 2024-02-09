from keras.models import Sequential
from keras.layers import (InputLayer, Dense, Conv2D, MaxPool2D, Dropout, Flatten)
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import os
import time

def criar_modelo() -> Sequential:
   seq = Sequential([
      InputLayer((28, 28, 1)),
      Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
      MaxPool2D((2, 2)),
      Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
      MaxPool2D((2, 2)),
      Flatten(),
      Dense(128, activation="relu"),
      Dropout(0.4),
      Dense(10, activation="softmax")
   ])

   seq.compile(
      SGD(0.01, 0.9),
      "categorical_crossentropy",
      metrics=['accuracy']
   )

   return seq

def calcular_tempos_camadas(modelo: Sequential, amostra):
   input_shape = modelo.input_shape[1:]  # Excluindo o batch_size
   input_amostra = np.random.rand(1, *input_shape)
   
   inicio = time.time()
   modelo.call(input_amostra)
   fim = time.time()
   t = (fim - inicio) * 1000

   print(f'tempo {t:.4f} ms')

def carregar_dados(n_treino: int=100, n_teste: int=100):
   (treino_x, treino_y), (teste_x, teste_y) = mnist.load_data()

   id_treino = np.random.choice(len(treino_x), n_treino, replace=False)
   treino_x = treino_x[id_treino]
   treino_y = treino_y[id_treino]

   id_teste = np.random.choice(len(teste_x), n_teste, replace=False)
   teste_x = teste_x[id_teste]
   teste_y = teste_y[id_teste]

   # normalizar os valores
   treino_x = treino_x.astype('float32') / 255.0
   teste_x = teste_x.astype('float32') / 255.0

   # converter para rótulos
   treino_y = to_categorical(treino_y, num_classes=10)
   teste_y = to_categorical(teste_y, num_classes=10)

   return treino_x, treino_y, teste_x, teste_y

def treinar_modelo(modelo: Sequential, treino_x, treino_y, epocas: int):
   modelo.fit(treino_x, treino_y, epochs=50, verbose=0)

if __name__ == '__main__':
   os.system('cls')

   modelo = criar_modelo()
   treino_x, treino_y, teste_x, teste_y = carregar_dados(1_000, 1_000)
   treinar_modelo(modelo, treino_x, treino_y, 50)

   # modelo.save("modelo-teste.keras")

   perda_treino, precisao_treino = modelo.evaluate(treino_x, treino_y)
   perda_teste,  precisao_teste  = modelo.evaluate(teste_x, teste_y)
   print("treino -> perda: ", perda_treino, " precisão: ", precisao_treino)
   print("teste -> perda: ", perda_teste, " precisão: ", precisao_teste)