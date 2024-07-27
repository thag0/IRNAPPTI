import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from modelos import ConvMnist
from analise_modelo import preparar_dataset

def get_activations(model, dataloader):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        model(inputs)
        break  # Run only one batch for efficiency
    
    for hook in hooks:
        hook.remove()

    return activations

def calculate_mutual_information(activations, labels):
    mutual_info = []
    labels = labels.detach().cpu().numpy()
    
    for activation in activations:
        # Flatten activation maps for MI calculation
        activation = activation.reshape(activation.shape[0], -1)
        mi = mutual_info_regression(activation, labels, discrete_features=True)
        mutual_info.append(mi.mean())
    
    return mutual_info

if __name__ == '__main__':
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    device = 'cpu'
    model = ConvMnist(device=device)
    model.eval()

    dl_treino, dl_teste = preparar_dataset('mnist')

    # Get activations from training set
    activations = get_activations(model, dl_treino)
    labels = torch.cat([y for _, y in dl_treino], dim=0)
    
    # Calculate mutual information
    mutual_info = calculate_mutual_information(activations, labels)

    # Plot the information plane
    plt.plot(range(len(mutual_info)), mutual_info, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Mutual Information')
    plt.title('Information Plane')
    plt.show()
