from keras.models import Sequential
from keras.layers import (InputLayer, Dense, Conv2D, ConvLSTM2D,MaxPool2D, TimeDistributed, Dropout, Flatten)
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.activations import leaky_relu
import numpy as np
import os

def criar_modelo_convolucional() -> Sequential:
   """
      Modelos com camadas convolucionais
   """

   modelo = Sequential([
      InputLayer((28, 28, 1)),
      Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
      MaxPool2D((2, 2)),
      Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
      MaxPool2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(10, activation="softmax")
   ])

   modelo.compile(
      SGD(0.01, 0.9),
      "categorical_crossentropy",
      metrics=['accuracy']
   )

   return modelo

def criar_modelo_convolucional_lstm() -> Sequential:
   """
      Modelos usando camadas convolucionais com memória
   """

   seq = Sequential([
      InputLayer((None, 28, 28, 1)),
      ConvLSTM2D(filters=32, kernel_size=(3, 3), activation='relu', return_sequences=True),
      TimeDistributed(MaxPool2D((2, 2))),
      ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu'),
      MaxPool2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(10, activation="softmax")
   ])

   seq.compile(
      SGD(0.01, 0.9),
      "categorical_crossentropy",
      metrics=['accuracy']
   )

   return seq

def criar_modelo_mlp() -> Sequential:
   """
      Modelos com apenas camadas densas
   """

   seq = Sequential([
      InputLayer((28, 28, 1)),
      Flatten(),
      Dense(28, activation='sigmoid'),
      Dense(28, activation='sigmoid'),
      Dense(10, activation="softmax")
   ])

   seq.compile(
      "adam",
      "categorical_crossentropy",
      metrics=['accuracy']
   )

   return seq

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
   historico = modelo.fit(treino_x, treino_y, epochs=epocas, verbose=0)
   return historico

if __name__ == '__main__':
   os.system('cls')


   epochs = 50

   treino_x, treino_y, teste_x, teste_y = carregar_dados(1_000, 1_000)

   # mlp
   modelo_mlp = criar_modelo_mlp()
   historico_mlp = treinar_modelo(modelo_mlp,  treino_x, treino_y, epochs)
   perda_teste, precisao_teste = modelo_mlp.evaluate(teste_x, teste_y)
   print("Mlp -> perda: ", perda_teste, " precisão: ", precisao_teste)
   
   # conv
   modelo_conv = criar_modelo_convolucional()
   historico_conv = treinar_modelo(modelo_conv, treino_x, treino_y, epochs)
   perda_teste,  precisao_teste  = modelo_conv.evaluate(teste_x, teste_y)
   print("Conv -> perda: ", perda_teste, " precisão: ", precisao_teste)
   
   # lstm
   modelo_lstm = criar_modelo_convolucional_lstm()
   treino_x_lstm = np.expand_dims(treino_x, axis=-1)  # Adiciona a dimensão do canal
   treino_x_lstm = np.expand_dims(treino_x_lstm, axis=1)  # Adiciona a dimensão da sequência de tempo
   teste_x_lstm = np.expand_dims(teste_x, axis=-1)  # Adiciona a dimensão do canal
   teste_x_lstm = np.expand_dims(teste_x_lstm, axis=1)  # Adiciona a dimensão da sequência de tempo
   historico_lstm = treinar_modelo(modelo_lstm, treino_x_lstm, treino_y, epochs)
   perda_teste,  precisao_teste  = modelo_lstm.evaluate(teste_x_lstm, teste_y)
   print("Lstm -> perda: ", perda_teste, " precisão: ", precisao_teste)