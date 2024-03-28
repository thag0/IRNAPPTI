import os
from keras.models import Sequential, load_model
from keras.preprocessing import image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.core.multiarray import ndarray
import tensorflow as tf
from keras.layers import (Conv2D, MaxPool2D)
from keras.datasets import cifar10
from keras.utils import to_categorical 

def carregar_modelo(caminho: str) -> Sequential:
   return load_model(caminho)

def carregar_imagem(caminho: str):
   img = image.load_img(caminho, target_size=(28, 28), color_mode='grayscale')
   img_tensor = image.img_to_array(img)
   img_tensor = np.expand_dims(img_tensor, axis=0)
   img_tensor /= 255.

   return img_tensor

def carregar_dados(n_treino: int=100, n_teste: int=100):
   (treino_x, treino_y), (teste_x, teste_y) = cifar10.load_data()

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

   return treino_x, treino_y, teste_x, teste_y

def entropia_condicional(previsoes) -> float:
   """
      Calcula o valor de incerteza do modelo em relação as sua previsões.
      
      Valores mais baixos indicam menor incerteza do modelo, que significa
      que o modelo tem bastante "confiança" na previsão feita.
   """

   ec = -tf.reduce_sum(previsoes * tf.math.log(previsoes + 1e-10), axis=-1)
   return float(ec)

def plotar_ativacoes(modelo: Sequential, entrada: ndarray, id_camada: int):
   if not isinstance(modelo.layers[id_camada], (Conv2D, MaxPool2D)):
      print("Id deve ser de uma camada convolucional ou maxpooling mas é de ", type(modelo.layers[id_camada]))
      return
   
   # pegar saída
   saida = entrada
   for i in range(len(modelo.layers)):
      saida = modelo.layers[i].call(saida)
      if i == id_camada:
         break

   if len(saida.shape) == 4:
      # o keras usa (batch, largura, altura, canais)
      # só tirando o batch_size (largura, altura, canais)
      saida = saida[0]

   # poltar saidas
   num_filtros = saida.shape[-1]
   n_col = 8
   n_lin = int(np.ceil(num_filtros / n_col))

   arr = list()
   val_max = 5
   for i in range(0, val_max):
      cor = i * (1 / val_max)
      arr.append((0, cor, cor))
      #gradiente roxo (cor, 0, cor)
      #gradiente azul (0, cor, cor)
   color_map = ListedColormap(arr)

   _, axs = plt.subplots(n_lin, n_col, figsize=(10, 6))
   for i in range(num_filtros):
      ax = axs[i // n_col, i % n_col]
      imagem = saida[..., i]
      ax.imshow(imagem, cmap='viridis')
 
      # remover os eixos X e Y do plot
      ax.set_xticks([])
      ax.set_yticks([])

   plt.tight_layout()
   plt.show()

def maior_indice(tensor) -> int:
   return np.argmax(tensor)

def testar_previsao(modelo: Sequential, amostra):
   saida: tf.Tensor = modelo.call(amostra)
   saida = tf.reshape(saida, (10, 1))
   print(saida)
   print("Previsto: ", maior_indice(saida))

if __name__ == '__main__':
   os.system('cls')

   modelo = carregar_modelo('./modelos/keras/modelo-cifar10.keras')
   _, _, teste_x, teste_y = carregar_dados(n_treino=2, n_teste=200)

   print('teste_x = ', teste_x.shape) #teste_x =  (200, 32, 32, 3)
   print('teste_y = ', teste_y.shape) #teste_y =  (200, 10)

   id_amostra = 2
   plt.imshow(teste_x[id_amostra])
   amostra = np.expand_dims(teste_x[id_amostra], axis=0)

   testar_previsao(modelo, amostra)

   plotar_ativacoes(modelo, amostra, 0)
   