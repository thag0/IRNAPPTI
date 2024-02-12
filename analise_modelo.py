import os
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras.activations import relu
from keras.layers import Conv2D
import matplotlib.pyplot as plt
import numpy as np

def carregar_modelo(caminho: str) -> Sequential:
   return load_model(caminho)

def carregar_imagem(caminho: str):
   img = image.load_img(caminho, target_size=(28, 28), color_mode='grayscale')
   img_tensor = image.img_to_array(img)
   img_tensor = np.expand_dims(img_tensor, axis=0)
   img_tensor /= 255.

   return img_tensor

def plotar_ativacoes(conv_layer: Conv2D, amostra):
   modelo = Sequential([conv_layer])
   predicoes = modelo.predict(amostra, verbose=0)

   num_filtros = predicoes.shape[-1]
   linhas = int(np.ceil(num_filtros / 8))
   _, axes = plt.subplots(linhas, 8, figsize=(14, 2 * linhas))
   for i, ax_row in enumerate(axes):
      for j, ax in enumerate(ax_row):
         if i * 8 + j < num_filtros:
            ax.imshow(predicoes[0, :, :, i * 8 + j], cmap='viridis')
         ax.axis('off')

   plt.show()

if __name__ == '__main__':
   os.system('cls')

   modelo = carregar_modelo('./modelos/modelo-teste.keras')
   conv1 = modelo.layers[0]

   amostra = carregar_imagem('./mnist/teste/8/img_0.jpg')
   conv1.call(amostra)
   # plotar_ativacoes(conv1, amostra)
