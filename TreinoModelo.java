import java.awt.image.BufferedImage;
import java.text.DecimalFormat;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import ged.Dados;
import ged.Ged;
import geim.Geim;
import jnn.Funcional;
import jnn.camadas.Conv2D;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;
import jnn.camadas.Flatten;
import jnn.camadas.MaxPool2D;
import jnn.core.tensor.Tensor;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;

public class TreinoModelo {
	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static Funcional jnn = new Funcional();

	// dados de controle
	static final int NUM_DIGITOS_TREINO = 10;
	static final int NUM_DIGITOS_TESTE  = NUM_DIGITOS_TREINO;
	static final int NUM_AMOSTRAS_TREINO = 500;//max 400
	static final int NUM_AMOSTRAS_TESTE  = 100;//max 100
	static final int TREINO_EPOCAS = 8;
	static final int TREINO_LOTE = 32;
	static final boolean TREINO_LOGS = true;

	// caminhos de arquivos externos
	static final String CAMINHO_TREINO = "./mnist/treino/";
	static final String CAMINHO_TESTE = "./mnist/teste/";
	static final String CAMINHO_SAIDA_MODELO = "./modelos/jnn/modelo-treinado.nn";
	static final String CAMINHO_HISTORICO = "historico-perda";

	public static void main(String[] args) {
		ged.limparConsole();
		
		final Tensor[] treinoX = jnn.arrayParaTensores(carregarDadosMNIST(CAMINHO_TREINO, NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO));
		final Tensor[] treinoY = jnn.arrayParaTensores(criarRotulosMNIST(NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO));

		Sequencial modelo = modeloConv();
		modelo.setHistorico(true);
		modelo.print();

		System.out.println("Treinando.");
		long tempo = System.nanoTime();
			modelo.treinar(treinoX, treinoY, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
		tempo = System.nanoTime() - tempo;

		long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
		long horas = segundosTotais / 3600;
		long minutos = (segundosTotais % 3600) / 60;
		long segundos = segundosTotais % 60;

		System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");
		System.out.print("Treino -> perda: " + modelo.avaliar(treinoX, treinoY).item() + " - ");
		System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(treinoX, treinoY).item() * 100), 4) + "%");

		System.out.println("\nCarregando dados de teste.");
		final var testeX = jnn.arrayParaTensores(carregarDadosMNIST(CAMINHO_TESTE, NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE));
		final var testeY = jnn.arrayParaTensores(criarRotulosMNIST(NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE));
		System.out.print("Teste -> perda: " + modelo.avaliar(testeX, testeY).item() + " - ");
		System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(testeX, testeY).item() * 100), 4) + "%");

		salvarModelo(modelo, CAMINHO_SAIDA_MODELO);
	}

	/*
	 * Criação de modelos Convolucionais.
	 */
	static Sequencial modeloConv() {
		Sequencial modelo = new Sequencial(
			new Entrada(1, 28, 28),
			new Conv2D(18, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Conv2D(22, new int[]{3, 3}, "relu"),
			new MaxPool2D(new int[]{2, 2}),
			new Flatten(),
			new Densa(90, "relu"),
			new Dropout(0.3),
			new Densa(NUM_DIGITOS_TREINO, "softmax")
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;
	}

	/*
	 * Criação de modelos Multilayer Perceptrons.
	 */
	static Sequencial modeloMlp() {
		Sequencial modelo = new Sequencial(
			new Entrada(1, 28, 28),
			new Flatten(),
			new Densa(12, "relu"),
			new Densa(12, "relu"),
			new Densa(NUM_DIGITOS_TREINO, "softmax")
		);

		modelo.compilar("adam", "entropia-cruzada");
		
		return modelo;
	}

	/**
	 * Salva o modelo num arquivo separado
	 * @param modelo modelo desejado.
	 * @param caminho caminho de destino
	 */
	static void salvarModelo(Sequencial modelo, String caminho) {
		String tipo = "float";
		System.out.println("Salvando modelo (" + tipo + ").");
		new Serializador().salvar(modelo, caminho, tipo);
	}

	/**
	 * Converte uma imagem numa matriz contendo seus valores de brilho entre 0 e 1.
	 * @param caminho caminho da imagem.
	 * @return matriz contendo os valores de brilho da imagem.
	 */
	static double[][] imagemParaMatriz(String caminho) {
		BufferedImage img = geim.lerImagem(caminho);
  
		double[][] imagem = new double[img.getHeight()][img.getWidth()];
		int[][] cinza = geim.obterCinza(img);
  
		for (int y = 0; y < imagem.length; y++) {
			for (int x = 0; x < imagem[y].length; x++) {
				imagem[y][x] = (double) cinza[y][x] / 255;
			}
		}
  
		return imagem;
  	}

	/**
	 * Carrega as imagens do conjunto de dados {@code MNIST}.
	 * <p>
	 *    Nota
	 * </p>
	 * O diretório deve conter subdiretórios, cada um contendo o conjunto de 
	 * imagens de cada dígito, exemplo:
	 * <pre>
	 *"mnist/treino/0"
	 *"mnist/treino/1"
	 *"mnist/treino/2"
	 *"mnist/treino/3"
	 *"mnist/treino/4"
	 *"mnist/treino/5"
	 *"mnist/treino/6"
	 *"mnist/treino/7"
	 *"mnist/treino/8"
	 *"mnist/treino/9"
	 * </pre>
	 * @param caminho caminho do diretório das imagens.
	 * @param amostras quantidade de amostras por dígito
	 * @param digitos quantidade de dígitos, iniciando do dígito 0.
	 * @return dados carregados.
	 */
	static double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos) {
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
							double[][] imagem = imagemParaMatriz(caminhoCompleto);
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
	static double[][] criarRotulosMNIST(int amostras, int digitos) {
		double[][] rotulos = new double[digitos * amostras][digitos];
		for (int numero = 0; numero < digitos; numero++){
			for (int i = 0; i < amostras; i++) {
				int indice = numero * amostras + i;
				rotulos[indice][numero] = 1;
			}
		}
		
		System.out.println("Rótulos gerados de 0 a " + (digitos-1) + ".");
		return rotulos;
	}

	/**
	 * Formata o valor recebido para a quantidade de casas após o ponto
	 * flutuante.
	 * @param valor valor alvo.
	 * @param casas quantidade de casas após o ponto flutuante.
	 * @return
	 */
	static String formatarDecimal(double valor, int casas) {
		String valorFormatado = "";

		String formato = "#.";
		for (int i = 0; i < casas; i++) formato += "#";

		DecimalFormat df = new DecimalFormat(formato);
		valorFormatado = df.format(valor);

		return valorFormatado;
	}

	/**
	 * Salva um arquivo csv com o historico de desempenho do modelo.
	 * @param modelo modelo.
	 * @param caminho caminho onde será salvo o arquivo.
	 */
	static void exportarHistorico(Modelo modelo, String caminho) {
		System.out.println("Exportando histórico de perda");
		double[] perdas = modelo.hist();
		double[][] dadosPerdas = new double[perdas.length][1];

		for (int i = 0; i < dadosPerdas.length; i++) {
			dadosPerdas[i][0] = perdas[i];
		}

		Dados dados = new Dados(dadosPerdas);
		ged.exportarCsv(dados, caminho);
	}
}
