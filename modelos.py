import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ConvMnist(nn.Module):
	def __init__(self, device: torch.device = None):
		"""
			Modelo convolucional usado para testes com dataset MNIST.

			device: dispositivo onde o modelo estará sendo usado (cpu ou gpu)
		"""
		super(ConvMnist, self).__init__()

		self.device = device if device is not None else 'cpu'

		self.model = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.MaxPool2d(kernel_size=(2, 2)),
			nn.Flatten(),
			nn.Linear(32 *5*5, 128),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(128, 10),
			nn.LogSoftmax(dim=1)
		)

		self.otimizador = optim.Adam(self.parameters(), lr=0.001)
		self.perda = nn.CrossEntropyLoss()

		self.to(self.device) # tentar usar na gpu para acelerar

	def get_camadas(self) -> list[nn.Module]:
		camadas = list(self.model.children())
		return camadas

	def forward(self, x) -> torch.Tensor:
		"""
			Alimenta os dados de entrada através do modelo.
			Os dados devem estar no mesmo dispositivo que o modelo (device).

			x: dados de entrada.
		"""

		# # Obter as dimensões após a camada de Flatten
		# batch_size = x.size(0)
		# flattened_dim = x.view(batch_size, -1).size(1)

		# # Ajustar a camada Linear para usar a dimensão correta
		# self.model[-3] = nn.Linear(flattened_dim, 128)

		return self.model.forward(x)

	def treinar(self, treino: DataLoader, epocas: int) -> list:
		"""
			Treina o modelo usando o conjunto de dados especificado.

			treino: conjunto de dados de treino.
			epocas: quantidade de épocas de treinamentos (passagens pelo dataset inteiro).

			Return: lista contendo os valores de perda durante o treino.
		"""

		self.train(True) #modo treino
		historico = []
		print('Treinando')

		for epoch in range(epocas):
			print(f"Epoch {epoch+1}/{epocas}")

			for batch in treino:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				prev = self.forward(x)
				val_perda =  self.perda.forward(prev, y)

				#backpropagation
				self.otimizador.zero_grad()
				val_perda.backward()
				self.otimizador.step()

				historico.append(val_perda.item())

		self.train(False)

		return historico

	def avaliar(self, teste: DataLoader) -> dict:
		"""
			Calula dados para métrica de desempenho do modelo.

			teste: conjunto de dados de teste.

			return: dicionário contendo o valor de perda e precisão do modelo.
		"""

		self.eval() # modo avaliação
		perda_total = 0.0
		precisao_total = 0.0

		for imgs, rotulos in teste:
			imgs, rotulos = imgs.to(self.device), rotulos.to(self.device)
			prev = self.forward(imgs)

			perda = self.perda.forward(prev, rotulos)
			perda_total += perda.item()

			precisao = (prev.argmax(dim=1) == rotulos).float().mean()
			precisao_total += precisao.item()
		
		perda_media = perda_total / len(teste)
		precisao_media = precisao_total / len(teste)
		
		return {
			'loss': perda_media,
			'accuracy': precisao_media
		}

	def salvar(self, caminho: str):
		"""
			Exporta o modelo num aquivo externo.
		"""

		self.to('cpu') # garantia
		torch.save(obj=self.state_dict(), f=caminho)
		print(f'Modelo salvo em \'{caminho}\'')

	def generate_gradcam(self, x, target_class):
		"""
		Gera o GradCAM para uma imagem de entrada.

		Args:
			x (torch.Tensor): A imagem de entrada.
			target_class (int): A classe alvo para a qual queremos visualizar a ativação.

		Returns:
			torch.Tensor: O GradCAM para a imagem de entrada.
		"""
		# Obter as camadas convolucionais
		conv_layers = [layer for layer in self.model if isinstance(layer, nn.Conv2d)]
		
		# Forward pass
		features = x.clone()
		for layer in conv_layers:
			features = layer(features)
			if layer != conv_layers[-1]:
					features = F.relu(features)

		# Calcular o gradiente da saída em relação às ativações
		self.zero_grad()
		target = torch.tensor([target_class], dtype=torch.long, device=x.device)
		one_hot_output = torch.zeros_like(self.forward(x))
		one_hot_output.scatter_(1, target.unsqueeze(1), 1.0)
		one_hot_output.requires_grad_(True)
		
		prev_layer_output = features.detach().requires_grad_(True)
		gradient = torch.autograd.grad(outputs=one_hot_output, inputs=prev_layer_output, grad_outputs=torch.ones_like(one_hot_output), create_graph=True, allow_unused=True)[0]

		# Verificar se o gradiente é None ou vazio
		if gradient is None or gradient.nelement() == 0:
			# Se o gradiente for None ou vazio, retornar um GradCAM vazio
			return torch.zeros_like(features)

		# Calcular os pesos do GradCAM
		weights = F.adaptive_avg_pool2d(gradient, 1)
		
		# Agregar os mapas de ativação ponderados
		gradcam = torch.zeros_like(features)
		for i, weight in enumerate(weights.squeeze()):
			gradcam[:, i] = weight * features[:, i]

		gradcam = gradcam.sum(dim=1, keepdim=True)
		gradcam = F.relu(gradcam)

		# Redimensionar e normalizar os mapas de ativação
		gradcam = F.interpolate(gradcam, size=x.shape[2:], mode='bilinear', align_corners=False)
		gradcam = gradcam - gradcam.min()
		gradcam = gradcam / gradcam.max()

		return gradcam


class ConvCifar10(nn.Module):
	def __init__(self, device: torch.device = None):
		"""
			Modelo convolucional usado para testes com dataset CIFAR-10.

			device: dispositivo onde o modelo estará sendo usado (cpu ou gpu)
		"""
		super(ConvCifar10, self).__init__()

		self.device = device if device is not None else 'cpu'

		self.model = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
 
			nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
 
			nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1, stride=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
 
			nn.Flatten(), 
			nn.Linear(256*4*4, 1024),
			nn.ReLU(),
			nn.Dropout(0.3),
			
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(0.3),

			nn.Linear(512, 10),
			nn.LogSoftmax(dim=1)
		)

		self.otimizador = optim.Adam(self.parameters(), lr=0.001)
		self.perda = nn.CrossEntropyLoss()

		self.to(self.device) # tentar usar na gpu para acelerar

	def get_camadas(self) -> list[nn.Module]:
		camadas = list(self.model.children())
		return camadas

	def forward(self, x) -> torch.Tensor:
		"""
			Alimenta os dados de entrada através do modelo.
			Os dados devem estar no mesmo dispositivo que o modelo (device).

			x: dados de entrada.
		"""

		return self.model.forward(x)

	def treinar(self, treino: DataLoader, epocas: int) -> list:
		"""
			Treina o modelo usando o conjunto de dados especificado.

			treino: conjunto de dados de treino.
			epocas: quantidade de épocas de treinamentos (passagens pelo dataset inteiro).

			Return: lista contendo os valores de perda durante o treino.
		"""

		self.train(True) #modo treino
		historico = []
		print('Treinando')

		for epoch in range(epocas):
			print(f"Epoch {epoch+1}/{epocas}")

			for batch in treino:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				prev = self.forward(x)
				val_perda =  self.perda.forward(prev, y)

				#backpropagation
				self.otimizador.zero_grad()
				val_perda.backward()
				self.otimizador.step()

				historico.append(val_perda.item())

		self.train(False)

		return historico

	def avaliar(self, teste: DataLoader) -> dict:
		"""
			Calula dados para métrica de desempenho do modelo.

			teste: conjunto de dados de teste.

			return: dicionário contendo o valor de perda e precisão do modelo.
		"""

		self.eval() # modo avaliação
		perda_total = 0.0
		precisao_total = 0.0

		for imgs, rotulos in teste:
			imgs, rotulos = imgs.to(self.device), rotulos.to(self.device)
			prev = self.forward(imgs)

			perda = self.perda.forward(prev, rotulos)
			perda_total += perda.item()

			precisao = (prev.argmax(dim=1) == rotulos).float().mean()
			precisao_total += precisao.item()
		
		perda_media = perda_total / len(teste)
		precisao_media = precisao_total / len(teste)
		
		return {
			'loss': perda_media,
			'accuracy': precisao_media
		}

	def salvar(self, caminho: str):
		"""
			Exporta o modelo num aquivo externo.
		"""

		torch.save(obj=self.state_dict(), f=caminho)
		print(f'Modelo salvo em \'{caminho}\'')

	def get_camadas(self) -> list[nn.Module]:
		camadas = list(self.model.children())
		return camadas

	def forward(self, x) -> torch.Tensor:
		"""
			Alimenta os dados de entrada através do modelo.
			Os dados devem estar no mesmo dispositivo que o modelo (device).

			x: dados de entrada.
		"""

		return self.model.forward(x)

	def treinar(self, treino: DataLoader, epocas: int) -> list:
		"""
			Treina o modelo usando o conjunto de dados especificado.

			treino: conjunto de dados de treino.
			epocas: quantidade de épocas de treinamentos (passagens pelo dataset inteiro).

			Return: lista contendo os valores de perda durante o treino.
		"""

		self.train(True) #modo treino
		historico = []
		print('Treinando')

		for epoch in range(epocas):
			print(f"Epoch {epoch+1}/{epocas}")

			for batch in treino:
				x, y = batch
				x, y = x.to(self.device), y.to(self.device)
				prev = self.forward(x)
				val_perda =  self.perda.forward(prev, y)

				#backpropagation
				self.otimizador.zero_grad()
				val_perda.backward()
				self.otimizador.step()

				historico.append(val_perda.item())

		self.train(False)

		return historico

	def avaliar(self, teste: DataLoader) -> dict:
		"""
			Calula dados para métrica de desempenho do modelo.

			teste: conjunto de dados de teste.

			return: dicionário contendo o valor de perda e precisão do modelo.
		"""

		self.eval() # modo avaliação
		perda_total = 0.0
		precisao_total = 0.0

		for imgs, rotulos in teste:
			imgs, rotulos = imgs.to(self.device), rotulos.to(self.device)
			prev = self.forward(imgs)

			perda = self.perda.forward(prev, rotulos)
			perda_total += perda.item()

			precisao = (prev.argmax(dim=1) == rotulos).float().mean()
			precisao_total += precisao.item()
		
		perda_media = perda_total / len(teste)
		precisao_media = precisao_total / len(teste)
		
		return {
			'loss': perda_media,
			'accuracy': precisao_media
		}

	def salvar(self, caminho: str):
		"""
			Exporta o modelo num aquivo externo.
		"""

		self.to('cpu') # garantia
		torch.save(obj=self.state_dict(), f=caminho)
		print(f'Modelo salvo em \'{caminho}\'')