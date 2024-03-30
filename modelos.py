import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Conv_pytorch(nn.Module):
   def __init__(self, entrada: tuple, device: torch.device):
      """
         Modelo convolucional usado para testes

         entrada: tupla contendo o formato de entrada (prof, alt, larg)
         device: dispositivo onde o modelo estará sendo usado (cpu ou gpu)
      """
      super(Conv_pytorch, self).__init__()

      self.device = device if device is not None else 'cpu'

      prof, alt, larg = entrada[0], entrada[1], entrada[0]

      self.conv1 = nn.Conv2d(in_channels=prof, out_channels=32, kernel_size=3)
      self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32, kernel_size=3)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.fc1 = nn.Linear(32 * 5 * 5, 128)
      self.fc2 = nn.Linear(self.fc1.out_features, 10)

      self.otimizador = optim.Adam(self.parameters(), lr=0.001)
      self.perda = nn.CrossEntropyLoss()
      
      self.to(self.device) # tentar usar na gpu para acelerar

   def forward(self, x):
      """
         Alimenta os dados de entrada através do modelo.
         
         x: dados de entrada.
      """

      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x))
      x = self.pool(x)
      x = x.view(-1, 32 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)

      return F.softmax(x, dim=1)
   
   def treinar(self, treino: DataLoader, epochs: int) -> list:
      self.train() #modo treino
      historico = []
      print('Treinando')

      for epoch in range(epochs):
         print(f"Epoch {epoch+1}/{epochs}")

         for i, data in enumerate(treino, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # Zero the parameter gradients
            self.otimizador.zero_grad()

            # Forward and backward pass
            outputs = self(inputs)
            loss = self.perda.forward(outputs, labels)
            historico.append(loss.item())
            loss.backward()

            # Update weights
            self.otimizador.step()

      return historico

   def avaliar(self, teste: DataLoader) -> dict:
      """
         Calula dados para métrica de desempenho do modelo.

         teste: conjunto de dados de teste.

         return: dicionário contendo o valor de perda e precisão do modelo. 
      """

      self.eval() # modo avaliação
      val_perda = 0.0
      acertos = 0
      total = 0

      with torch.no_grad():
         for data in teste:
            entradas, saidas = data[0].to(self.device), data[1].to(self.device)

            outputs = self(entradas)
            loss = self.perda(outputs, saidas)
            val_perda += loss.item()
            _, previsto = torch.max(outputs.data, 1)
            total += saidas.size(0)
            acertos += (previsto == saidas).sum().item()

      res = {}
      res['loss'] = val_perda / len(teste)
      res['accuracy'] = 100 * (acertos / total)
      return res
   
   def salvar(self, caminho: str):
      """
         Exporta o modelo num aquivo externo.
      """

      torch.save(obj=self.state_dict(), f=caminho)
      print('Modelo salvo')