import torch
from modelos import Conv_pytorch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

def plotar_historico(historico: list):
   plt.plot(historico, label='Perda de Treinamento')
   plt.xlabel('Época')
   plt.ylabel('Perda')
   plt.title('Histórico de Perda do Modelo')
   plt.legend()
   plt.ylim(bottom=0)
   plt.show()

def preparar_dataset() -> tuple[DataLoader]:
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

   # Carregar os datasets de treinamento e teste
   train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

   # Criar DataLoaders para carregamento em lote
   tam_lote: int = 128
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=tam_lote, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=tam_lote, shuffle=False)

   return (train_loader, test_loader)

if __name__ == '__main__':
   os.system('cls')

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   modelo = Conv_pytorch(1, device=device)
   print(modelo)
   print(f"Modelo em: {modelo.device}")

   dl_treino, dl_teste = preparar_dataset()

   hist = modelo.treinar(dl_treino, 3)

   metricas_treino = modelo.avaliar(dl_treino)
   metricas_teste = modelo.avaliar(dl_teste)

   print('treino:', metricas_treino)
   print('teste:', metricas_teste)

   plotar_historico(hist)

   modelo.salvar("./modelos/pytorch/conv-pytorch-mnist.pt")