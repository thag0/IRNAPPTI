import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the model
class ConvMnist(nn.Module):
    def __init__(self, device: torch.device = None):
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
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

        self.otimizador = optim.Adam(self.parameters(), lr=0.001)
        self.perda = nn.CrossEntropyLoss()

        self.to(self.device)

    def forward(self, x) -> torch.Tensor:
        return self.model.forward(x)

    def treinar(self, treino: DataLoader, epocas: int, save_activations=False):
        self.train(True)
        historico = []
        activations_list = []

        for epoch in range(epocas):
            for batch in treino:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                prev = self.forward(x)
                val_perda = self.perda(prev, y)

                self.otimizador.zero_grad()
                val_perda.backward()
                self.otimizador.step()

                historico.append(val_perda.item())

                # Save activations if required
                if save_activations:
                    activations = self.get_activations(x)
                    activations_list.append(activations)

        self.train(False)
        return historico, activations_list

    def get_activations(self, x):
        activations = []
        current_input = x
        for layer in self.model:
            current_input = layer(current_input)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                activations.append(current_input.clone().detach().cpu().numpy())
        return activations

def distribution_from_bins(bins):
    # bins in N^(n times l)
    _, counts = np.unique(bins, axis=0, return_counts=True)
    # counts in N^n
    return counts/sum(counts)

def entropy(X):
    dX = distribution_from_bins(X)
    return -np.sum(dX * np.log2(dX))

def conditional_entropy(X,Y):
    ys, ycnt = np.unique(Y, return_counts=True)
    cond_entropy = 0
    for y,cnt in zip(ys,ycnt):
        cond_entropy += entropy(X[Y==y])*(cnt/len(Y))
    return cond_entropy

def Mutual_Info(Y, X):
    return entropy(X)-conditional_entropy(X,Y)

a = torch.tensor([0.1, 0.2, 0.1, 0.4, 0.2])
b = torch.tensor([0.2, 0.2, 0.3, 0.2, 0.1])

print(Mutual_Info(a, b))

# Dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Instancia o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = ConvMnist(device)

# Treina o modelo e salva ativações
EPOCHS = 4
_, activations_list = model.treinar(train_loader, EPOCHS, save_activations=True)

# Verificando o conteúdo das ativações antes da discretização
if len(activations_list) == 0:
    raise ValueError("activations_list está vazio antes da discretização.")

for i, activations in enumerate(activations_list):
    print(f"Epoch {i}: Número de camadas capturadas: {len(activations)}")
    for j, layer_activations in enumerate(activations):
        print(f"    Camada {j}: Forma das ativações: {layer_activations.shape}")

# Verificando formas das ativações
for i, activations in enumerate(activations_list):
    print(f"Epoch {i}")
    for j, layer_activations in enumerate(activations):
        print(f"    Layer {j} shape: {layer_activations.shape}")

for epoch_activations in activations_list:
    for layer_activations in epoch_activations:
        print(f"Shape before flattening: {layer_activations.shape}")
        flattened_activations = layer_activations.reshape(-1)
        print(f"Number of elements after flattening: {len(flattened_activations)}")

def discretization(activations_list, bins, batch_size=16):
    discretized_activations = []
    
    for epoch_activations in activations_list:
        discretized_epoch = []
        for layer_activations in epoch_activations:
            flattened_activations = layer_activations.reshape(-1).astype(np.float16)
            bins_values = np.linspace(np.min(flattened_activations), np.max(flattened_activations), bins + 1)
            
            discretized_parts = []
            num_samples = flattened_activations.shape[0]
            
            for i in range(0, flattened_activations.shape[0], batch_size):  # Processar por lote
                part_activations = flattened_activations[i:i+batch_size]
                discretized_part = np.digitize(part_activations, bins_values)
                discretized_parts.append(discretized_part)

            # Concatenate discretized parts of the same layer
            if discretized_parts:
                discretized_epoch.append(np.concatenate(discretized_parts, axis=0).reshape(layer_activations.shape))
        
        # Append the entire epoch's discretized activations
        if discretized_epoch:
            discretized_activations.append(discretized_epoch)
    
    return discretized_activations


discretized_activations=discretization(activations_list, bins=30)

# Verificando o conteúdo de discretized_activations
if len(discretized_activations) == 0:
    raise ValueError("discretized_activations está vazio.")
    
for i, activations in enumerate(discretized_activations):
    print(f"Epoch {i}: Número de camadas discretizadas: {len(activations)}")
    for j, layer_activations in enumerate(activations):
        print(f"    Camada {j}: Forma dos dados discretizados: {layer_activations.shape}")


def information_plane(X, Y, discretized_activations, EPOCHS):
    I_XT = np.zeros((len(discretized_activations[0]), EPOCHS))
    I_TY = np.zeros((len(discretized_activations[0]), EPOCHS))

    for epoch in range(EPOCHS):
        for layer in range(len(discretized_activations[epoch])):
            I_XT[layer, epoch] = Mutual_Info(discretized_activations[epoch][layer], X)
            I_TY[layer, epoch] = Mutual_Info(discretized_activations[epoch][layer], Y)
    
    return I_XT, I_TY

# Pegando um subconjunto dos dados de treino
X_subset, y_subset = next(iter(train_loader))
X_subset = X_subset.view(-1, 28*28).numpy()
y_subset = y_subset.numpy()

information_plane_values = information_plane(X_subset, y_subset, discretized_activations, EPOCHS)

import matplotlib.pyplot as plt

def plot_information_plane_custom(IXT_array, ITY_array, num_epochs, every_n):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel(r'$I(X;T)$')
    ax.set_ylabel(r'$I(T;Y)$')
    ax.set_title('Information Plane')

    # Cores para cada camada
    layer_colors = plt.cm.get_cmap('viridis', IXT_array.shape[0])

    def plot_point(i):
        for j in range(len(IXT_array[:, 0])):
            ax.plot(IXT_array[j, (i-1):(i+1)], ITY_array[j, (i-1):(i+1)], '.-', 
                    c=layer_colors(j), alpha=0.6, ms=8)
        ax.plot(IXT_array[:, i], ITY_array[:, i], 'k-', alpha=0.2)

    for i in range(0, num_epochs, every_n):
        plot_point(i)

    plt.show()

    return fig

plot_information_plane_custom(information_plane_values[0], information_plane_values[1], num_epochs=EPOCHS, every_n=1)

def plot_logs(train_logs, test_logs, size=(10, 6)):
    plt.figure(1, figsize=size)

    # Ordena e separa os dados dos logs de treinamento
    train_epochs, train_accuracies = zip(*sorted(train_logs.items()))
    plt.plot(train_epochs, train_accuracies, label='Treino', color='blue')

    # Ordena e separa os dados dos logs de teste
    test_epochs, test_accuracies = zip(*sorted(test_logs.items()))
    plt.plot(test_epochs, test_accuracies, label='Teste', color='green')

    plt.ylabel('Precisao')
    plt.xlabel('Numero da Epoca')
    plt.legend()
    plt.title('Precisao vs. Numero da Epoca')
    plt.show()

# Exemplo de como usar a função
train_logs = {1: 0.8, 2: 0.85, 3: 0.87, 4: 0.9}
test_logs = {1: 0.75, 2: 0.8, 3: 0.83, 4: 0.85}

plot_logs(train_logs, test_logs)