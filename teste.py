import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from modelos import ConvMnist
from analise_modelo import preparar_dataset

def hook_fn(module, input, output):
    activations.append(output)

def calculate_mutual_information(X, Y):
    X_flattened = X.flatten()
    Y_flattened = Y.flatten()
    
    # Verificar se os vetores têm o mesmo comprimento
    if len(X_flattened) != len(Y_flattened):
        min_len = min(len(X_flattened), len(Y_flattened))
        X_flattened = X_flattened[:min_len]
        Y_flattened = Y_flattened[:min_len]
    
    # Calcular a informação mútua
    mi = mutual_info_regression(X_flattened.reshape(-1, 1), Y_flattened)
    return mi[0]  # Retornar o valor da informação mútua

def plot_mutual_information(layers, mi_values):
    plt.figure(figsize=(10, 6))
    plt.plot(layers, mi_values, marker='o')
    plt.xlabel('Camadas')
    plt.ylabel('Informação Mútua')
    plt.title('Informação Mútua entre Camadas')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    import os
    os.system('cls')

    device = 'cpu'
    model = ConvMnist(device=device)
    model.eval()

    activations = []
    layer_names = []

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            layer_names.append(name)
            layer.register_forward_hook(hook_fn)

    dl_treino, dl_teste = preparar_dataset('mnist')

    for lote in dl_teste:
        x, y = lote
        x = x.to(device)
        model(x)
        break

    mi_values = []
    for i in range(len(activations) - 1):
        activation1 = activations[i].detach().cpu().numpy()
        activation2 = activations[i + 1].detach().cpu().numpy()

        mi = calculate_mutual_information(activation1, activation2)
        mi_values.append(mi)

    plot_mutual_information(layer_names[:-1], mi_values)
