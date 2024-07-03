import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Definir uma rede neural simples
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Função de perda do Information Bottleneck
def information_bottleneck_loss(activations, targets, beta):
    # Suponha que as ativações sejam uma lista de tensores para cada camada
    # Calcule os termos de informação mútua (versão simplificada)
    I_XT = sum([torch.mean(torch.square(act)) for act in activations])  # Termo de exemplo
    I_TY = -nn.CrossEntropyLoss()(activations[-1], targets)  # Termo de exemplo
    
    loss = I_XT - beta * I_TY
    return loss

# Função para treinar o modelo e coletar ativações e perdas
def train_model(model, train_loader, optimizer, beta, epochs=5):
    losses = []
    activations_history = []

    for epoch in range(epochs):
        for data, targets in train_loader:
            optimizer.zero_grad()
            output = model(data)
            activations = [data.view(-1, 784), torch.relu(model.fc1(data.view(-1, 784))), torch.relu(model.fc2(torch.relu(model.fc1(data.view(-1, 784)))))]
            loss = information_bottleneck_loss(activations, targets, beta)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            activations_history.append([act.detach().cpu().numpy() for act in activations])

    return losses, activations_history

# Carregar conjunto de dados (MNIST ou Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Inicializar o modelo, otimizador e parâmetros
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
beta = 0.1

# Treinar o modelo
losses, activations_history = train_model(model, train_loader, optimizer, beta, epochs=5)

# Função para plotar a evolução das perdas e ativações
def plot_information_bottleneck(losses, activations_history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plotar a perda de Information Bottleneck
    axs[0].plot(losses)
    axs[0].set_title('Information Bottleneck Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')

    # Plotar as ativações das camadas
    activation_means = [np.mean([np.mean(act) for act in epoch]) for epoch in activations_history]
    axs[1].plot(activation_means)
    axs[1].set_title('Mean Activation Values Across Layers')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Mean Activation')

    plt.tight_layout()
    plt.show()

# Plotar os resultados
plot_information_bottleneck(losses, activations_history)
