# -*- coding: utf-8 -*-

import torch
import torch.utils
import torch.utils.tensorboard
from modelos import (ConvMnist, ConvCifar10)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from torchmetrics.clustering import mutual_info_score

from torch.utils.tensorboard import SummaryWriter

def plotar_historico(historico: list):
	plt.plot(historico, label='Perda de Treinamento')
	plt.xlabel('Época')
	plt.ylabel('Perda')
	plt.title('Histórico de Perda do Modelo')
	plt.legend()
	plt.ylim(bottom=0)
	plt.show()

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

def get_sample_batch(dl: DataLoader):
    batch = next(iter(dl))
    input_data, target = batch
    return input_data, target

def add_img_grid(writer: SummaryWriter, imgs, titulo: str):
	grid = torchvision.utils.make_grid(imgs, nrow=4, pad_value=2)
	writer.add_image(titulo, grid)

if __name__ == '__main__':
	os.system('cls')

	writer = SummaryWriter('/runs/mnist')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	modelo = ConvMnist(device=device)
	print(modelo)
	print(f"Modelo em: {modelo.device}")

	dl_treino, dl_teste = preparar_dataset('mnist')
	epocas = 6

	# add amostras do dataset
	batch_samples, batch_labels = get_sample_batch(dl_teste)
	samples_grid = batch_samples[:8]
	add_img_grid(writer, samples_grid, 'Amostras de teste')

	# add grafico do modelo
	example_input = torch.rand(1, 1, 28, 28).to(device)
	writer.add_graph(modelo, example_input)

	hist = modelo.treinar(dl_treino, epocas)

	# resultado de perda
	for i, registro in enumerate(hist):
		writer.add_scalar('Training Loss', registro['loss'], global_step=i)
		writer.add_scalar('Training Accuracy', registro['accuracy'], global_step=i)

	metricas_treino = modelo.avaliar(dl_treino)
	metricas_teste = modelo.avaliar(dl_teste)

	print('treino:', metricas_treino)
	print('teste:', metricas_teste)

	plotar_historico(hist)

	modelo.salvar("./modelos/pytorch/conv-pytorch-mnist.pt")

	writer.close()