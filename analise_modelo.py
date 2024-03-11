import os
from keras.models import Sequential, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import ndarray
import tensorflow as tf

def carregar_modelo(caminho: str) -> Sequential:
   return load_model(caminho)

def carregar_imagem(caminho: str):
   img = image.load_img(caminho, target_size=(28, 28), color_mode='grayscale')
   img_tensor = image.img_to_array(img)
   img_tensor = np.expand_dims(img_tensor, axis=0)
   img_tensor /= 255.

   return img_tensor

def plotar_ativacoes(modelo: Sequential, amostra: ndarray, id_camada: int):
   predicoes = amostra
   for i in range(len(modelo.layers)):
      camada = modelo.layers[i]
      predicoes = camada.call(predicoes)

      if(i == id_camada):
         break

   print(predicoes.shape)

def entropia_condicional(previsoes) -> float:
   """
      Calcula o valor de incerteza do modelo em relação as sua previsões.
      
      Valores mais baixos indicam menor incerteza do modelo, que significa
      que o modelo tem bastante "confiança" na previsão feita.
   """

   ec = -tf.reduce_sum(previsoes * tf.math.log(previsoes + 1e-10), axis=-1)
   return float(ec)

if __name__ == '__main__':
   os.system('cls')

   modelo = carregar_modelo('./modelos/keras/modelo-teste.keras')
   amostra = carregar_imagem('./mnist/teste/4/img_0.jpg')
   plotar_ativacoes(modelo, amostra, 0)