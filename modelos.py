import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv_pytorch(nn.Module):
   def __init__(self, entrada: tuple):
      super(Conv_pytorch, self).__init__()
      prof, alt, larg = entrada[0], entrada[1], entrada[0]

      self.conv1 = nn.Conv2d(in_channels=prof, out_channels=32, kernel_size=3)
      self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
      self.fc1 = nn.Linear(32 * 5 * 5, 128)
      self.fc2 = nn.Linear(128, 10)

   def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x))
      x = self.pool(x)
      x = x.view(-1, 32 * 5 * 5)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return F.softmax(x, dim=1)
   
   def train_model(self, train_loader, epochs=5):
      optimizer = optim.Adam(self.parameters(), lr=0.001)
      criterion = nn.CrossEntropyLoss()

      for epoch in range(epochs):
         running_loss = 0.0
         for i, data in enumerate(train_loader, 0):
               inputs, labels = data

               optimizer.zero_grad()

               outputs = self.forward(inputs)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()

               running_loss += loss.item()
               if i % 100 == 99:  # Imprimir a cada 100 mini-lotes
                  print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 100))
                  running_loss = 0.0