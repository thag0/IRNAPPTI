from keras.models import Sequential
from keras.layers import (InputLayer, Dense, Conv2D, ConvLSTM2D,MaxPool2D, TimeDistributed, Dropout, Flatten)
from keras.optimizers import SGD
from keras.datasets import (mnist, cifar10)
from keras.utils import to_categorical
from keras.activations import leaky_relu
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

import torch
from modelos import Conv_pytorch
import torchvision
import torchvision.transforms as transforms


def criar_modelo_convolucional() -> Sequential:
   """
      Modelos com camadas convolucionais
   """

   modelo = Sequential([
      InputLayer((28, 28, 1)),
      Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
      MaxPool2D((2, 2)),
      Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
      MaxPool2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(10, activation='softmax')
   ])

   modelo.compile(
      "adam",
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

   # Selecionar um subconjunto aleatório de dados
   treino_x = treino_x[:n_treino]
   treino_y = treino_y[:n_treino]
   teste_x = teste_x[:n_teste]
   teste_y = teste_y[:n_teste]

   # Normalizar os valores de pixel para o intervalo [0, 1]
   treino_x = treino_x.astype('float32') / 255.0
   teste_x = teste_x.astype('float32') / 255.0

   # Converter rótulos em formato one-hot
   treino_y = to_categorical(treino_y, num_classes=10)
   teste_y = to_categorical(teste_y, num_classes=10)
   print(treino_x.shape)

   return treino_x, treino_y, teste_x, teste_y

def treinar_modelo(modelo: Sequential, treino_x, treino_y, epocas: int, verbose=0):
   historico = modelo.fit(treino_x, treino_y, epochs=epocas, verbose=verbose)
   return historico

def plotar_historico(historico):
   # Plotar a perda do modelo
   plt.plot(historico.history['loss'], label='Perda de Treinamento')
   plt.xlabel('Época')
   plt.ylabel('Perda')
   plt.title('Histórico de Perda do Modelo')
   plt.legend()
   plt.ylim(bottom=0)
   plt.show()

if __name__ == '__main__':
   os.system('cls')

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(device)

   modelo = Conv_pytorch((1, 28, 28), device)
   print(modelo)

   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

   modelo.train_model(train_loader, 5)

   exit()

   epochs = 5

   treino_x, treino_y, teste_x, teste_y = carregar_dados(50_000, 10_000)
   
   # conv
   conv = criar_modelo_convolucional()
   historico_conv = treinar_modelo(conv, treino_x, treino_y, epochs, 1)
   
   perda, precisao  = conv.evaluate(treino_x, treino_y)
   print("(Treino) -> perda: ", perda, " precisão: ", precisao)
   
   perda, precisao  = conv.evaluate(teste_x, teste_y)
   print("(Teste) -> perda: ", perda, " precisão: ", precisao)

   plotar_historico(historico_conv)
   conv.save('modelos/keras/modelo-mnist.keras')