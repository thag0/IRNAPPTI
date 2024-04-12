import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.modules import (Conv2d, MaxPool2d)
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from modelos import ConvMnist, ConvCifar10
from PIL import Image
import torch.nn.functional as F

def carregar_modelo_mnist(device: torch.device, caminho: str) -> ConvMnist:
   modelo = ConvMnist('cpu')
   modelo.load_state_dict(torch.load(caminho))
   modelo.to(device)
   return modelo

def carregar_modelo_cifar(device: torch.device, caminho: str) -> ConvCifar10:
   modelo = ConvCifar10('cpu')
   modelo.load_state_dict(torch.load(caminho))
   modelo.to(device)
   return modelo

def carregar_imagem(caminho: str) -> torch.Tensor:
   img = Image.open(caminho)
   
   preprocess = transforms.Compose([
      transforms.ToTensor()
   ])
   
   # Aplicar as transformações na imagem
   tensor_img = preprocess(img).unsqueeze(0)  # Adicionar uma dimensão extra para o lote (batch)
   
   return tensor_img

def preparar_dataset(nome: str) -> tuple[DataLoader]:
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

   if nome == 'mnist':
      dataset = torchvision.datasets.MNIST
   
   elif nome == 'cifar-10':
      dataset = torchvision.datasets.CIFAR10

   else:
      print(f'Dataset \'{nome}\' não encontrado.')

   # Carregar os datasets de treinamento e teste
   train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
   test_dataset = dataset(root='./data', train=False, download=True, transform=transform)

   # Criar DataLoaders para carregamento em lote
   tam_lote: int = 128
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tam_lote, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=tam_lote, shuffle=False)

   return (train_loader, test_loader)

def plotar_ativacoes(modelo: ConvCifar10, entrada: torch.Tensor, id_camada: int):
   camadas = modelo.get_camadas()
   camada = camadas[id_camada]
   if not isinstance(camada, (Conv2d, MaxPool2d)):
      print("Id deve ser de uma camada convolucional ou maxpooling mas é de ", type(camada))
      return
   
   # # pegar saída
   saida = entrada.to(modelo.device)

   print(saida.device)
   print(modelo.device)
   
   for i in range(len(camadas)):
      saida = camadas[i].forward(saida)
      if i == id_camada:
         break

   # o pytorch usa (batch, canais, largura, altura)
   # só tirando o batch_size (canais, largura, altura)
   saida = saida.squeeze(0)

   # poltar saidas
   num_filtros = saida.shape[0]
   n_col = 8
   n_lin = int(np.ceil(num_filtros / n_col))

   arr = list()
   val_max = 10
   for i in range(0, val_max):
      cor = i * (1 / val_max)
      arr.append((0, cor, cor))
      #gradiente roxo (cor, 0, cor)
      #gradiente azul (0, cor, cor)
   color_map = ListedColormap(arr)

   _, axs = plt.subplots(n_lin, n_col, figsize=(10, 6))
   for i in range(num_filtros):
      ax = axs[i // n_col, i % n_col]
      s = saida.detach().cpu().numpy()
      imagem = s[i, ...]
      ax.imshow(imagem, cmap='viridis')
      #ax.imshow(imagem, cmap=color_map)
 
      # remover os eixos X e Y do plot
      ax.set_xticks([])
      ax.set_yticks([])

   plt.tight_layout()
   plt.show()

def testar_previsao(modelo: ConvCifar10, amostra: torch.Tensor):
   prev = modelo.forward(amostra)
   val = prev.argmax(dim=1).item()
   print(f'Previsto = {val}')

def obter_amostra(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
   for x, y in data_loader:
      return x[1], y[1]


if __name__ == '__main__':
   os.system('cls')

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   modelo = carregar_modelo_mnist(device, './modelos/pytorch/conv-pytorch-mnist.pt')
   dl_treino, dl_teste = preparar_dataset('mnist') #formato N C W H

   lote = next(iter(dl_teste))
   imgs, classes = lote

   id_img = 0
   img = imgs[id_img].unsqueeze(0).to(device)
   rotulo = classes[id_img]

   with torch.no_grad():
      saida = modelo(img)

   classe_alvo = rotulo.item()
   gradcam = modelo.generate_gradcam(img, classe_alvo)

   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.title('Imagem Original')
   plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
   plt.axis('off')

   plt.subplot(1, 2, 2)
   plt.title('GradCAM')
   # Soma dos gradientes ao longo do canal
   gradcam_sum = gradcam.sum(dim=1, keepdim=True)
   gradcam_sum = gradcam_sum.squeeze().cpu().numpy()
   print(gradcam_sum)
   plt.imshow(gradcam_sum, cmap='hot', interpolation='nearest')
   plt.axis('off')

   plt.show()

   # ----------------

   # amostra = carregar_imagem('./cifar/horse.png')

   # print(amostra.shape)

   # conv_ids = []

   # for i in range(len(modelo.get_camadas())):
   #    if isinstance(modelo.get_camadas()[i], Conv2d):
   #       conv_ids.append(i)

   # testar_previsao(modelo, amostra)
   # plotar_ativacoes(modelo, amostra, conv_ids[0])
   