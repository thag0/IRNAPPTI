# Interpretabilidade de Redes Neurais Profundas

Diretório destinado aos arquivos para testes dentro do projeto de pesquisa
sobre a interpretabilidade de Redes Neurais Artificiais Profundas na perspectiva da Teoria da Informação

O repositório contém alguns arquivos de testes usando o datase MNIST para criação dos primeiros modelos convolucionais que serão testados.

# Alguns métodos de análise incluem
- <strong> Entropia condicional </strong> : para avaliar a taxa de "certeza" da previsão de um modelo com base nos dados de saída gerados;

- <strong> Visualização de camadas </strong> : para mostrar quais característica estão sendo aprendidas por cada camada do modelo, usado para entender como o modelo processa informação;

- <strong> Visualização de gradientes </strong> : para mostrar quais pixels da imagem ajudam na classificação, usado para entender como o modelo toma decisões.

- <strong> GradCam </strong> : gerar mapas de calor para mostrar quais regiões de entrada são mais significativas para o modelo fazer sua previsão.