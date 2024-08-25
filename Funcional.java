import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import geim.Geim;
import geim.Pixel;
import render.JanelaImagem;
import render.JanelaDesenho;
import render.JanelaMatriz;
import jnn.camadas.Camada;
import jnn.camadas.Conv2D;
import jnn.core.tensor.Tensor;
import jnn.core.tensor.Variavel;
import jnn.modelos.Sequencial;
import jnn.core.Utils;

/**
 * Interface funcional.
 */
public class Funcional {

	/**
	 * Caminho das imagens de teste do mnist.
	 */
	static final String CAMINHO_IMAGEM = "./mnist/teste/";

	/**
	 * Gerenciador de imagens.
	 */
	private Geim geim = new Geim();

	/**
	 * Utilitário.
	 */
	private Utils utils = new Utils();

	/**
	 * Interface funcional.
	 */
	public Funcional() {}

	/**
	 * Calcula a entropia com os dados do tensor.
	 * <p>
	 *		A Entropia é uma medida matemática da quantidade de 
	 *		informação ou surpresa associada a um evento ou sistema.
	 * </p>
	 * <p>
	 *		Essa medida quantifica o grau de incerteza ou surpresa associado 
	 *		ao sistema. Quanto maior a entropia, maior a incerteza. A entropia é 
	 *		máxima quando todos os eventos são igualmente prováveis.
	 * </p>
	 * @param x {@code Tensor} base, com dados em formato de distribuição
	 * de probabilidade.
	 * @return {@code Tensor} resultado.
	 */
	public Tensor entropia(Tensor x) {
		Tensor res = x.clone();

		if (x.numDim() > 1) res.flatten();

		return res.map(v -> (v == 0) ? 0 : v * Math.log(v)/Math.log(2))
			.reduce(0.0, (x1, x2) -> x1 + x2)
			.mul(-1.0);
	}

	public Tensor entropiaCondicional(Tensor x, Tensor y) {
		Map<Double, List<Double>> conditionalDistributions = new HashMap<>();
		for (int i = 0; i < x.tam(); i++) {
			double xi = x.get(i);
			double yi = y.get(i);
			conditionalDistributions
				.computeIfAbsent(xi, k -> new ArrayList<>())
				.add(yi);
		}

		double total = x.tam();
		double conditionalEntropy = 0.0;
		for (Map.Entry<Double, List<Double>> entry : conditionalDistributions.entrySet()) {
			List<Double> yValues = entry.getValue();
			Map<Double, Integer> histogram = new HashMap<>();
			for (double yi : yValues) {
				histogram.put(yi, histogram.getOrDefault(yi, 0) + 1);
			}

			double subTotal = yValues.size();
			double entropy = 0.0;
			for (Map.Entry<Double, Integer> histEntry : histogram.entrySet()) {
				double prob = (double) histEntry.getValue() / subTotal;
				entropy += prob * Math.log(prob) / Math.log(2);
			}

			conditionalEntropy += (subTotal / total) * -entropy;
		}

		return new Tensor(new double[]{ conditionalEntropy }, 1);
	}

	public Tensor informacao_mutua(Tensor x, Tensor y) {
		return entropia(x).sub(entropiaCondicional(x, y));
	}

	/**
	 * Calcula o mapa de calor do Grad-CAM.
	 * @param modelo modelo treinado.
	 * @param entrada amostra de entrada.
	 * @param rotulo rótulo desejado.
	 * @return {@code Tensor} contendo o mapa de calor calculado.
	 */
	public Tensor gradCAM(Sequencial modelo, Tensor entrada, Tensor rotulo) {
		//pegar índice da última camada convolucional do modelo
		int idConv = -1;
		for (int i = 0; i < modelo.numCamadas(); i++) {
			if (modelo.camada(i) instanceof Conv2D) idConv = i;
		}

		if (idConv == -1) {
			throw new IllegalArgumentException(
				"\nNenhuma camada convolucional encontrada no modelo."
			);
		}

		//camada de interesse da análise
		Conv2D conv = (Conv2D) modelo.camada(idConv);

		// passo de backpropagation para ter os gradientes calculados
		Tensor prev = modelo.forward(entrada);
		Tensor grad = modelo.perda().derivada(prev, rotulo); 
		for (int i = modelo.numCamadas()-1; i >= 0; i--) {
			grad = modelo.camada(i).backward(grad);
		}
		
		// calcular mapa de calor
		Tensor convRes  = conv._somatorio.clone();// saída "bruta" da camada
		Tensor convGrad = conv._gradSaida.clone();
		int canais  = convGrad.shape()[0];
		int altura  = convGrad.shape()[1];
		int largura = convGrad.shape()[2];
	
		Tensor heatmap = new Tensor(altura, largura);

		for (int i = 0; i < canais; i++) {
			double alfa = convGrad.subTensor(i).media().item();
			heatmap.add(
				convRes.subTensor(i).map(x -> x*alfa)
			);
		}

		heatmap
		.relu()// preservar características que tem influência positiva na classe de interesse
		.norm(0.0, 1.0);// ajudar na visualização

		// redimensionar o mapa de calor para as dimensões da imagem de entrada
		int altEntrada  = entrada.shape()[1];
		int largEntrada = entrada.shape()[2];

		return redim2D(
			heatmap, altEntrada, largEntrada
		);
	}

	/**
	 * Desenha uma janela gráfica para testar o modelo treinado com o dataset
	 * {@code MNIST} em tempo real.
	 * @param modelo modelo treinado.
	 */
	public void desenharMnist(Sequencial modelo) {
		final byte fator = 28;
		final int escala = 18;

		new Thread(() -> {
			JanelaDesenho jd = new JanelaDesenho(fator*escala, fator*(escala*2), modelo);

			while (jd.isVisible()) {
				jd.update();
	
				try {
					Thread.sleep(90);
				} catch (Exception e) {}
			}
	
			jd.dispose();
		}).start();
	}

	/**
	 * Calcula a matriz de confusão das predições do modelo.
	 * @param modelo modelo treinado.
	 */
	public void matrizConfusao(Sequencial modelo, int amostras) {
		System.out.println("Calculando Matriz de Confusão");
		int digitos = 10;
		Tensor[] x = utils.arrayParaTensores(carregarDadosMNIST(CAMINHO_IMAGEM, amostras, digitos));
		Tensor[] y = utils.arrayParaTensores(criarRotulosMNIST(amostras, digitos));
		
		Tensor m = modelo.avaliador().matrizConfusao(x, y);

		new Thread(() -> {
			JanelaMatriz jm = new JanelaMatriz(500, 500, m);
			jm.exibir();
		}).start();
	}

	/**
	 * Carrega os dados de entrada do MNIST (apenas features).
	 * @param amostras quantidade de amostras por dígito
	 * @param digitos quantidade de dígitos, iniciando do dígito 0.
	 * @return dados carregados.
	 */
	double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos) {
		final double[][][][] imagens = new double[digitos * amostras][1][][];
		final int numThreads = Runtime.getRuntime().availableProcessors() / 2;
  
		try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
			int id = 0;
			for (int i = 0; i < digitos; i++) {
				for (int j = 0; j < amostras; j++) {
					final String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
					final int indice = id;
					
					exec.submit(() -> {
						try {
							double[][] imagem = carregarImagemCinza(caminhoCompleto);
							imagens[indice][0] = imagem;
						} catch (Exception e) {
							System.out.println(e.getMessage());
							System.exit(1);
						}
					});

					id++;
				}
			}
  
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
  
		System.out.println("Imagens carregadas (" + imagens.length + ").");
  
		return imagens;
	}

	/**
	 * Carrega os dados de saída do MNIST (classes / rótulos)
	 * @param amostras quantidade de amostras por dígito
	 * @param digitos quantidade de dígitos, iniciando do dígito 0.
	 * @return dados carregados.
	 */
	double[][] criarRotulosMNIST(int amostras, int digitos) {
		double[][] rotulos = new double[digitos * amostras][digitos];
		
		for (int dig = 0; dig < digitos; dig++) {
			for (int ams = 0; ams < amostras; ams++) {
				int indice = dig * amostras + ams;
				rotulos[indice][dig] = 1;
			}
		}
		
		return rotulos;
	}

	/**
	 * Abre uma janela gráfica contendo a saída da camada fornecida.
	 * <p>
	 *    É necessário que a camada tenha pré calculado algum resultado para
	 *    que ele poda ser visualizado.
	 * </p>
	 * @param conv camada convolucional.
	 * @param escala escala de ampliação da imagem original.
	 * @param norm normaliza os valores entre 0 e 1 para evitar artefatos
	 * na janela gráfica.
	 */
	public void desenharSaidas(Camada conv, Tensor amostra, int escala, boolean norm) {
		Conv2D camada = null;

		try {
			camada = (Conv2D) conv;
		} catch (Exception e) {
			throw new IllegalArgumentException(
				"\nCamada não é do tipo Convolucional e sim " + 
				camada.getClass().getSimpleName() + ", escolha uma camada Conv2D."
			);
		}

		Tensor prev = camada.forward(amostra);
		int filtros = camada.numFiltros();
		Tensor[] arr = new Tensor[filtros];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = prev.subTensor(i);
		}
		
		desenharImagens(arr, escala, norm, true, "Saidas Conv");
	}

	/**
	 * Desenha o conteúdo do tensor em forma de imagem com uma janela gráfica.
	 * @param tensor tensor com os dados desejados.
	 * @param escala escala de ampliação da janela.
	 * @param norm normalizar os valores do tensor entre 0 e 1
	 * @param titulo nome da janela.
	 */
	public void desenharImagem(Tensor tensor, int escala, boolean norm, String titulo) {
		if (tensor.numDim() != 2 && tensor.numDim() != 3) {
			throw new IllegalArgumentException(
				"\nTensor deve ser 2D ou 3D, mas é " + tensor.numDim() + "D."
			);
		}

		int[] shape = tensor.shape();
		int altura  = shape[shape.length-2];
		int largura = shape[shape.length-1];
		JanelaImagem janela = new JanelaImagem(altura, largura, escala, titulo);

		if (norm) {
			Tensor clone = tensor.clone().norm(0.0, 1.0);
			janela.desenharImagem(clone);

		} else {
			janela.desenharImagem(tensor);
		}	
	}

	/**
	 * Desenha o conteúdo dos tensores em forma de imagem, com uma janela gráfica.
	 * @param arr array de tensores.
	 * @param escala escala de ampliação da janela.
	 * @param norm normalizar os valores entre 0 e 1.
	 */
	public void desenharImagens(Tensor[] arr, int escala, boolean norm, boolean inplace, String titulo) {
		int[] shape = arr[0].shape();
		int[] dim = {
			shape[shape.length-2],// altura
			shape[shape.length-1] // largura
		};

		JanelaImagem janela = new JanelaImagem(dim[0], dim[1], escala, titulo);

		if (norm) {
			Tensor[] ts = new Tensor[arr.length];
			if (inplace) {
				for (int i = 0; i < arr.length; i++) {
					ts[i] = arr[i].norm(1.0, 0.0);
				}

			} else {//não alterar dados recebidos
				for (int i = 0; i < arr.length; i++) {
					ts[i] = arr[i].clone().norm(1.0, 0.0);
				}
			}

			janela.desenharImagens(arr);
		
		} else {
			janela.desenharImagens(arr);
		}

	}

	/**
	 * Salva os resultados das ativações e pré ativações de uma camada 
	 * convolucional do modelo
	 * @param modelo modelo desejado.
	 * @param idConv índice da camada convolucional do modelo.
	 * @param norm normaliza os valores entre 0 e 1.
	 */
	public void exportarAtivacoes(Sequencial modelo, int idConv, boolean norm, int escala) {
		Conv2D camada;
		try {
			camada = (Conv2D) modelo.camada(idConv);
		} catch (Exception e) {
			throw new IllegalArgumentException(
				"\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
				modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
			);
		}

		String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");

		final int digitos = 10;
		int[] shape = camada.saida().shape();
		final int canais = shape[0];
		for (int i = 0; i < digitos; i++) {
			String caminhoAmostra = CAMINHO_IMAGEM + i + "/img_0.jpg";
			Tensor amostra = new Tensor(carregarImagemCinza(caminhoAmostra));
			amostra.unsqueeze(0);// 2D -> 3D
			modelo.forward(amostra);

			Tensor[] somatorios = new Tensor[canais];
			Tensor[] saidas = new Tensor[canais];

			for (int j = 0; j < saidas.length; j++) {
				Tensor tempSaida = new Tensor(camada.saida().subTensor(j));
				Tensor tempSomatorio = new Tensor(camada._somatorio.subTensor(j));
				
				if (norm) {
					tempSaida.norm(0, 1);
					tempSomatorio.norm(0, 1);
				}
				
				saidas[j]     = new Tensor(tempSaida);
				somatorios[j] = new Tensor(tempSomatorio);
			}

			String caminhoSomatorio = "./resultados/pre-ativacoes/" + diretorioCamada + "/" + i + "/";
			String caminhoSaida = "./resultados/ativacoes/" + diretorioCamada + "/" + i + "/";

			limparDiretorio(caminhoSomatorio);
			limparDiretorio(caminhoSaida);

			exportarImagens(saidas, escala, caminhoSaida);
			exportarImagens(somatorios, escala, caminhoSomatorio);
		}

		System.out.println("Ativações exportadas para a camada " + idConv);
	}

	/**
	 * Exporta os filtros da camada convolucional
	 * @param modelo modelo desejado.
	 * @param idConv índice da camada convolucional do modelo.
	 * @param norm normaliza os valores entre 0 e 1.
	 */
	public void exportarFiltros(Sequencial modelo, int idConv, boolean norm, int escala) {
		Conv2D camada;
		try {
			camada = (Conv2D) modelo.camada(idConv);
		} catch (Exception e) {
			throw new IllegalArgumentException(
				"\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
				modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
			);
		}

		String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");
		String caminho = "./resultados/filtros/" + diretorioCamada + "/";

		Tensor filtros = camada._kernel;
		limparDiretorio(caminho);

		int[] shapeFiltro = filtros.shape();
		int numFiltros = shapeFiltro[0];
		int altFiltro = shapeFiltro[2];
		int largFiltro = shapeFiltro[3];
		Tensor[] arrFiltros = new Tensor[numFiltros];
		for (int i = 0; i < numFiltros; i++) {
			Tensor slice = filtros.slice(new int[]{i, 0, 0, 0}, new int[]{i+1, 1, altFiltro, largFiltro});
			slice.squeeze(0).squeeze(0);// 4d -> 2d

			Tensor temp = new Tensor(slice);
			if (norm) temp.norm(0, 1);
			arrFiltros[i] = temp;
		}

		exportarImagens(arrFiltros, escala, caminho);

		System.out.println("Filtros exportados para a camada " + idConv);
	}

	/**
	 * Salva os valores das matrizes como imagens no caminho especificado.
	 * @param arr array de matrizes.
	 * @param caminho diretório onde os arquivos serão salvos.
	 */
	public void exportarImagens(Tensor[] arr, int escala, String caminho) {
		if (arr == null) {
			throw new IllegalArgumentException(
				"\nArray fornecido é nulo."
			);
		}

		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == null) {
				throw new IllegalArgumentException(
					"\nElemento " + i + " do array é nulo."
				);      
			}
		}

		for (int i = 0; i < arr.length; i++) {
			exportarImagem(arr[i], (caminho + "amostra-" + (i+1)), escala);
		}
	}

	/**
	 * Salva a matriz num arquivo de imagem externo.
	 * @param img matriz desejada.
	 * @param caminho diretório de destino.
	 * @param escala escala de tratamento da imagem final.
	 */
	public void exportarImagem(Tensor img, String caminho, double escala) {
		if (img.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTensor deve ser 2D"
			);
		}

		int[] shape = img.shape();
		int altura =  (int) (shape[0] * escala);
		int largura = (int) (shape[1] * escala);
		Pixel[][] estrutura = new Pixel[altura][largura];
  
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				int originalY = (int) (y / escala);
				int originalX = (int) (x / escala);

				double cinza = img.get(originalY, originalX);
				int c = (int) (cinza * 255);
				estrutura[y][x] = new Pixel(c, c, c);
			}
		}

		File diretorio = new File(caminho).getParentFile();
		if (!diretorio.exists()) diretorio.mkdirs();
  
		geim.exportarPng(estrutura, caminho);
	}

	/**
	 * Carrega a imagem a partir de um arquivo.
	 * @param caminho caminho da imagem.
	 * @return {@code Tensor} contendo os dados da imagem no
	 * padrão RGB.
	 */
	public Tensor carregarImagemRGB(String caminho) {
		BufferedImage img = geim.lerImagem(caminho);
		int altura = img.getHeight(), largura = img.getWidth();

		Tensor imagem = new Tensor(3, altura, largura);

		int[][] r = geim.obterVermelho(img);
		int[][] g = geim.obterVerde(img);
		int[][] b = geim.obterAzul(img);

		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				imagem.set(((double)(r[y][x]) / 255), 0, y, x);
				imagem.set(((double)(g[y][x]) / 255), 1, y, x);
				imagem.set(((double)(b[y][x]) / 255), 2, y, x);
			}
		}

		return imagem;
	}

	/**
	 * Carrega a imagem a partir de um arquivo.
	 * @param caminho caminho da imagem.
	 * @return matriz contendo os dados da imagem em
	 * escala de cinza.
	 */
	public double[][] carregarImagemCinza(String caminho) {
		BufferedImage bImg = geim.lerImagem(caminho);
		int altura = bImg.getHeight(), largura = bImg.getWidth();
		
		int[][] cinza = geim.obterCinza(bImg);   
		
		double[][] img = new double[altura][largura];
		for (int y = 0; y < altura; y++) {
			for (int x = 0; x < largura; x++) {
				img[y][x] = ((double) cinza[y][x]) / 255.0;
			}
		}

		return img;
	}

	/**
	 * Limpa os arquivos do diretório.
	 * @param caminho caminho do diretório.
	 */
	public void limparDiretorio(String caminho) {
		File diretorio = new File(caminho);
  
		if (diretorio.isDirectory()) {
			File[] arquivos = diretorio.listFiles();
  
			if (arquivos != null) {
				for (File arquivo : arquivos) {
					if (arquivo.isFile()) arquivo.delete();
				}
			}

		} else {
			System.out.println("\nO caminho fornecido (" + caminho + ") não é um diretório válido.");
		}
	}

	/**
	 * Calcula o índice que contém o maior valor no array.
	 * @param arr array base.
	 * @return índice com o maior valor.
	 */
	public int maiorIndice(double[] arr) {
		int id = 0;
		double maior = arr[0];

		for (int i = 1; i < arr.length; i++) {
			if (arr[i] > maior) {
				id = i;
				maior = arr[i];
			}
		}

		return id;
	}

	/**
	 * Calcula o índice que contém o maior valor no array.
	 * @param arr array base.
	 * @return índice com o maior valor.
	 */
	public int maiorIndice(Variavel[] arr) {
		int id = 0;
		double maior = arr[0].get();

		for (int i = 1; i < arr.length; i++) {
			if (arr[i].get() > maior) {
				id = i;
				maior = arr[i].get();
			}
		}

		return id;
	}

	/**
	 * Auxiliar para gerar um dítigo baseado no conjunto de dados do MNIST.
	 * <p>
	 *    Exemplo: 
	 * </p>
	 * <pre>
	 *double[] arr = gerarRotuloMnist(2);
	 *arr = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
	 * </pre>
	 * @param digito digito desejado de {@code 0 a 9}
	 * @return array contendo a saída categórica para o índice desejado.
	 */
	public double[] gerarRotuloMnist(int digito) {
		double[] arr = new double[10];

		for (int i = 0; i < arr.length; i++) {
			arr[i] = 0.0;
		}
		arr[digito] = 1.0d;

		return arr;
	}

	/**
	 * Interpola os valores da matriz para o novo tamanho.
	 * @param tensor matriz desejada.
	 * @param novaAlt novo valor de altura da matriz.
	 * @param novaLarg novo valor de largura da matriz.
	 * @return {@code Tensor} redimensionado.
	 */
	public Tensor redim2D(Tensor tensor, int novaAlt, int novaLarg) {
		if (tensor.numDim() != 2) {
			throw new IllegalArgumentException("\nTensor deve ser 2D.");
		}

		int alt  = tensor.shape()[0];
		int larg = tensor.shape()[1];

		Tensor redim = new Tensor(novaAlt, novaLarg);
		
		for (int i = 0; i < novaAlt; i++) {
			for (int j = 0; j < novaLarg; j++) {
				double ampAlt  = (double) i / (novaAlt - 1)  * (alt - 1);
				double ampLarg = (double) j / (novaLarg - 1) * (larg - 1);
				
				int y0 = (int) ampAlt;
				int x0 = (int) ampLarg;
				int y1 = Math.min(y0 + 1, alt - 1);
				int x1 = Math.min(x0 + 1, larg - 1);
				
				double dx = ampLarg - x0;
				double dy = ampAlt - y0;
				
				double val = (
					(1 - dx) * (1 - dy) * tensor.get(y0, x0) +
					dx * (1 - dy) * tensor.get(y0, x1) +
					(1 - dx) * dy * tensor.get(y1, x0) +
					dx * dy * tensor.get(y1, x1)
				);
				
				redim.set(val, i, j);
			}
		}
		
		return redim;
	}
}
