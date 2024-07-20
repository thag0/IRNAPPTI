import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.api.layers

# Binarização e Ruído Gaussiano
def g(x):
    return 0 if x < 0 else 1

def f(x):
    y = 100 * np.cos(np.sum(x, axis=1)) - np.sum(x, axis=1)**2 + 5 * np.random.normal(0, 1, 10000)
    return np.array([g(i) for i in y])

# Criando X e Y
x = np.array([[random.randint(0, 1) for _ in range(10)] for _ in range(10000)])
y = f(x).reshape(-1, 1)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
input_shape = len(X_train[0])
print(X_train.shape)
print(y_train.shape)

# Funções de entropia e informação mútua
def entropy(Y):
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    return np.sum(-prob * np.log2(prob))

def jEntropy(Y, X):
    YX = np.c_[Y, X]
    return entropy(YX)

def cEntropy(Y, X):
    return jEntropy(Y, X) - entropy(X)

def Mutual_Info(Y, X):
    return entropy(Y) - cEntropy(Y, X)

# Modelo de rede neural
layers = [3, 2, 1]
EPOCHS = 100
BATCH_SIZE = 180

model = Sequential([
    Input(shape=(10,)),
    Dense(layers[0], activation='tanh'),
    Dense(layers[1], activation='tanh'),
    Dense(layers[2], activation='sigmoid')
])

model.compile(optimizer='SGD', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Callback para salvar ativações
activations_list = []

def save_activations(model, X_train):
    functors = [tf.function(lambda x: layer(x)) for layer in model.layers]
    # Usando um lote de dados para passar para a função de callback
    batch_data = X_train[:BATCH_SIZE]
    layer_activations = [f(batch_data) for f in functors]
    activations_list.append(layer_activations)

activations_callback = LambdaCallback(on_epoch_end=lambda batch, logs: save_activations(model, X_train))

result = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[activations_callback])
loss, acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")

# Discretização dos valores contínuos das camadas
def discretization(activations_list, bins, layers, EPOCHS):
    n_bins = bins
    for layer in range(len(layers)):
        for epoch in range(EPOCHS):
            bins = np.linspace(
                np.min(activations_list[epoch][layer]),
                np.max(activations_list[epoch][layer]), 
                n_bins + 1
            )
            activations_list[epoch][layer] = np.digitize(activations_list[epoch][layer], bins)
    return activations_list

activations_list = discretization(activations_list, 30, layers, EPOCHS)

# Calculando I(X,T) e I(T,Y)
def information_plane(X, Y, activations_list, layers, EPOCHS):
    I_XT = np.zeros((len(layers), EPOCHS))
    I_TY = np.zeros((len(layers), EPOCHS))
    for layer in range(len(layers)):
        for epoch in range(EPOCHS):
            I_XT[layer, epoch] = Mutual_Info(activations_list[epoch][layer], X)
            I_TY[layer, epoch] = Mutual_Info(activations_list[epoch][layer], Y)
    return I_XT, I_TY

information_plane_values = information_plane(X_train, y_train, activations_list, layers, EPOCHS)

# Função para plotar o plano de informação
def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n, I_XY):
    assert len(IXT_array) == len(ITY_array)
    max_index = len(IXT_array)

    plt.figure(figsize=(12, 6), dpi=150)
    plt.xlabel(r'$I(X;T)$')
    plt.ylabel(r'$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]
    cmap_layer = plt.get_cmap('Greys')
    clayer = [cmap_layer(i) for i in np.linspace(0, 1, max_index)]

    for i in range(max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        plt.plot(IXT, ITY, color=clayer[i], linewidth=2, label='Layer {}'.format(i))
        plt.scatter(IXT, ITY, marker='o', c=colors, s=200, alpha=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')
    plt.axhline(y=I_XY, color='red', linestyle=':', label=r'$I[X,Y]$')
    plt.legend()
    plt.show()

I_XY = Mutual_Info(X_train, y_train)
plot_information_plane(information_plane_values[0], information_plane_values[1], EPOCHS, 1, I_XY)
