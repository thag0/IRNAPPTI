import os
from keras.models import Sequential, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray
import tensorflow as tf
from keras.layers import (Conv2D, MaxPool2D)

def carregar_modelo(caminho: str) -> Sequential:
   return load_model(caminho)

def carregar_imagem(caminho: str):
   img = image.load_img(caminho, target_size=(28, 28), color_mode='grayscale')
   img_tensor = image.img_to_array(img)
   img_tensor = np.expand_dims(img_tensor, axis=0)
   img_tensor /= 255.

   return img_tensor

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

   _, axs = plt.subplots(n_lin, n_col, figsize=(10, 6))
   for i in range(num_filtros):
      ax = axs[i // n_col, i % n_col]
      imagem = saida[..., i]
      ax.imshow(imagem, cmap="gray")
 
      # remover os eixos X e Y do plot
      ax.set_xticks([])
      ax.set_yticks([])

   plt.tight_layout()
   plt.show()

def maior_indice(tensor) -> int:
   return np.argmax(tensor)

if __name__ == '__main__':
   os.system('cls')

   modelo = carregar_modelo('./modelos/keras/modelo-teste.keras')
   amostra = carregar_imagem('./mnist/teste/4/img_0.jpg')
   # plotar_ativacoes(modelo, amostra, 0)
   
   saida: tf.Tensor = modelo.call(amostra)
   saida = tf.reshape(saida, (10, 1))
   print(saida)
   print("Previsto: ", maior_indice(saida))

   # teste