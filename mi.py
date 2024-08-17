import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

class MIHook:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()  # Use detach() para evitar rastreamento de gradientes
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def calculate_mi(self, input_data, target_data, target_layers):
        mi_results = {}
        for layer in target_layers:
            if layer in self.activations:
                T = self.activations[layer]
                IXT = informacao_mutua(input_data, T)
                ITY = informacao_mutua(T, target_data)
                mi_results[layer] = (IXT, ITY)
        return mi_results

def entropia(x: torch.Tensor) -> float:
    x = x.view(-1)
    prob_dist = torch.softmax(x, dim=0)  # Normaliza para obter uma distribuição de probabilidade
    return torch.sum(-prob_dist * torch.log(prob_dist + 1e-10)).item()  # Adicione .item() para retornar um float

def entropia_condicional(X: torch.Tensor, Y: torch.Tensor) -> float:
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    num_classes = int(Y_np.max() + 1)
    ec = 0.0

    if Y_np.ndim == 2:
        Y_np = Y_np.argmax(axis=1)

    for y in range(num_classes):
        mask = (Y_np == y)
        
        if mask.any():
            X_filtered = X_np[mask]
            ec += entropia(torch.tensor(X_filtered)) * (mask.sum() / len(Y_np))

    return ec

def informacao_mutua(x: torch.Tensor, y: torch.Tensor) -> float:
    return entropia(x) - entropia_condicional(x, y) 

def plano_informacao(modelo, dl_treino, epochs):
    mi_hook = MIHook(modelo)
    mi_results_history = []

    for epoch in range(epochs):
        modelo.train()
        for lote in dl_treino:
            x, y = lote
            x, y = x.to(modelo.device), y.to(modelo.device)

            _ = modelo(x)

        print(f'Época {epoch}/{epochs}')
        layers_of_interest = ['r1', 'r2', 'r3']
        mi_results = mi_hook.calculate_mi(x, y, layers_of_interest)
        mi_results_history.append(mi_results)

    mi_hook.remove_hooks()

    fig, ax = plt.subplots()

    for epoch_mi in mi_results_history:
        for layer, (i_xt, i_ty) in epoch_mi.items():
            ax.scatter(i_xt, i_ty, label=f'{layer} (Epoch)')

    ax.set_xlabel('I(X; T)')
    ax.set_ylabel('I(T; Y)')
    ax.set_title('Plano da Informação')
    plt.show()