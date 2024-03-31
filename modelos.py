import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Conv_pytorch(nn.Module):
   def __init__(self, prof: int, device: torch.device):
      """
         Modelo convolucional usado para testes

         prof: profundidade dos dados de entrada que serão usados
               no modelo. Exemplo: grayscale = 1, RGB = 3.
         device: dispositivo onde o modelo estará sendo usado (cpu ou gpu)
      """
      super(Conv_pytorch, self).__init__()

      self.device = device if device is not None else 'cpu'

      self.model = nn.Sequential(
         nn.Conv2d(in_channels=prof, out_channels=32, kernel_size=(3, 3)),
         nn.ReLU(),
         nn.Dropout(0.3),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
         nn.ReLU(),
         nn.Dropout(0.3),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Flatten(),
         nn.Linear((32 * 5*5), 128),
         nn.ReLU(),
         nn.Dropout(0.3),
         nn.Linear(128, 10),
         nn.LogSoftmax(dim=1)
      )

      self.otimizador = optim.Adam(self.parameters(), lr=0.001)
      self.perda = nn.CrossEntropyLoss()

      self.to(self.device) # tentar usar na gpu para acelerar

   def forward(self, x):
      """
         Alimenta os dados de entrada através do modelo.

         x: dados de entrada.
      """

      return self.model.forward(x)

   def treinar(self, treino: DataLoader, epochs: int) -> list:
      self.train() #modo treino
      historico = []
      print('Treinando')

      for epoch in range(epochs):
         print(f"Epoch {epoch+1}/{epochs}")

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