import torch
import numpy as np

def entropia(x: torch.Tensor):
    # Certifique-se de que x é um tensor 1D antes de aplicar a operação
    x = x.view(-1)
    return torch.sum(-x * torch.log(x + 1e-10))

def entropia_condicional(X: torch.Tensor, Y: torch.Tensor) -> float:
    X_np = X.detach().cpu().numpy()  # Use detach() para evitar rastreamento de gradientes
    Y_np = Y.detach().cpu().numpy()  # Use detach() para evitar rastreamento de gradientes

    num_classes = int(Y_np.max() + 1)  # Assegure-se de que `num_classes` é um inteiro
    ec = 0.0

    # Verifique se Y_np é 1D ou 2D
    if Y_np.ndim == 2:
        Y_np = Y_np.argmax(axis=1)  # Obtém a classe prevista

    for y in range(num_classes):
        # Crie uma máscara booleana para a classe `y`
        mask = (Y_np == y)
        
        # Verifique se há elementos na máscara
        if mask.any():
            # Filtra `X_np` usando a máscara, garantindo que `X_np` e a máscara tenham a mesma forma
            X_filtered = X_np[mask]
            
            # Calcule a entropia para o subconjunto filtrado
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
                # print(f"Hook para \'{name}\' ativado com shape: {output.shape}")
                self.activations[name] = output
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
                    print(f"Calculando MI para {layer1} e {layer2} com shapes x: {x.shape} e y: {y.shape}")
                    mi = informacao_mutua(x, y)
                    mi_results[(layer1, layer2)] = mi
                else:
                    print(f"Ativações para {layer1} ou {layer2} não encontradas.")

        return mi_results
