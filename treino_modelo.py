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
   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

   return (train_loader, test_loader)

if __name__ == '__main__':
   os.system('cls')

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   modelo = Conv_pytorch(entrada=(1, 28, 28), device=device)
   print(modelo)
   print(f"Modelo em: {modelo.device}")

   transform = transforms.Compose([
      transforms.ToTensor(), 
      transforms.Normalize((0.5,), (0.5,))
   ])

   mnist_treino = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   ds_treino = DataLoader(mnist_treino, batch_size=64, shuffle=True)

   mnist_teste = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
   ds_teste = DataLoader(mnist_teste, batch_size=64, shuffle=True)

   hist = modelo.treinar(ds_treino, 2)

   metricas_treino = modelo.avaliar(ds_treino)
   metricas_teste = modelo.avaliar(ds_teste)

   print('treino:', metricas_treino)
   print('teste:', metricas_teste)

   plotar_historico(hist)

   modelo.salvar("./modelos/pytorch/conv-pytorch-mnist.pt")