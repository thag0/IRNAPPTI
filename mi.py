import torch
import numpy as np

def entropia(x: torch.Tensor):
    x = x.view(-1)
    return torch.sum(-x * torch.log(x + 1e-10))

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
            ec += entropia(torch.tensor(X_filtered)).item() * (mask.sum() / len(Y_np))

    return ec

def informacao_mutua(x: torch.Tensor, y: torch.Tensor):
    return entropia(x) - entropia_condicional(x, y) 

class MIHook:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
                # print(f'Hook para \'{name}\' ativado com shape: {output.shape}')
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module):
                self.hooks.append(module.register_forward_hook(hook_fn(name)))
                print(f'hook adicionado para \'{name}\'')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def calculate_mi(self, target_layers):
        mi_results = {}
        for i, layer1 in enumerate(target_layers):
            for layer2 in target_layers[i+1:]:
                if layer1 in self.activations and layer2 in self.activations:
                    x = self.activations[layer1]
                    y = self.activations[layer2]
                    # print(f"Calculando MI para {layer1} e {layer2} com shapes x: {x.shape} e y: {y.shape}")
                    mi = informacao_mutua(x, y)
                    mi_results[(layer1, layer2)] = mi
                else:
                    print(f"Ativações para {layer1} ou {layer2} não encontradas.")

        return mi_results

