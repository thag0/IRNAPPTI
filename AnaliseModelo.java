import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ged.Ged;
import geim.Geim;
import jnn.core.tensor.OpTensor;
import jnn.core.tensor.Tensor;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;


public class AnaliseModelo {
	static final String CAMINHO_MODELO = "./modelos/jnn/";
	static final String CAMINHO_IMAGEM = "./mnist/teste/";

	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static OpTensor optensor = new OpTensor();
	static Funcional f = new Funcional();
	static Serializador serializador = new Serializador();
	
	public static void main(String[] args) {
		ged.limparConsole();

		// String nomeModelo = "conv-mnist-dropout";
		String nomeModelo = "conv-mnist-97-4";
		// String nomeModelo = "mlp-mnist-90";
		// String nomeModelo = "modelo-treinado";
		Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELO + nomeModelo + ".nn");
		// modelo.print();

		final int digito = 3;
		Tensor amostra = new Tensor(f.carregarImagemCinza(CAMINHO_IMAGEM +  digito + "/img_0.jpg"));
		amostra.unsqueeze(0);//2d -> 3d
		
		// Tensor rotulo = new Tensor(f.gerarRotuloMnist(digito), 10);
		// Tensor heatmap = f.gradCAM(modelo, amostra, rotulo);
		// Tensor heatpmapRGB = tensorCinzaParaRGB(heatmap);
		// Tensor amostraRGB = tensorCinzaParaRGB(amostra.clone().squeeze(0));

		// amostraRGB.aplicar(x -> x*0.95);
		// coresTensor(heatpmapRGB, 0.6, 0.2, 0.9);

		// f.desenharImagem(heatpmapRGB, 10, false, "Heatmap");
		// f.desenharImagem(amostraRGB, 10, false, "Amostra");
		// f.desenharImagem(amostraRGB.clone().add(heatpmapRGB), 10, false, "Heatmap + Amostra");
		
		// f.desenharMnist(modelo);

		f.matrizConfusao(modelo, 100);

		// f.desenharSaidas(modelo.camada(0), amostra, 15, true);

		// testarAcertosMNIST(modelo);

		// Tensor prev = modelo.forward(amostra);
		// double ec = f.entropiaCondicional(prev);
		// System.out.println("Entropia condicional: " + String.format("%.2f", (1-ec)*100));
		// prev.view(10, 1).print();
		// System.out.println("Dígito: " + digito + " -> Previsto: " + f.maiorIndice(prev.paraArray()));

		// new Thread(() -> {
		// 	boolean normalizar = true;
		// 	int escala = 20;
		// 	System.out.println("Exportando ativações...");
		// 	f.exportarAtivacoes(modelo, 0, normalizar, escala);
		// 	f.exportarFiltros(modelo, 0, normalizar, escala);
		// 	f.exportarAtivacoes(modelo, 2, normalizar, escala);
		// 	f.exportarFiltros(modelo, 2, normalizar, escala);
		// }).start();
	}

	/**
	 * Usa o modelo para prever todos os dados de teste.
	 * @param modelo modelo treinado.
	 */
	static void testarTodosDados(Sequencial modelo) {
		final int digitos = 10;
		final int amostras = 100;
		for (int i = 0; i < digitos; i++) {
			for (int j = 0; j < amostras; j++) {
				testarPrevisao(modelo, (i + "/img_" + j), false);
			}
			System.out.println();
		}
	}

	/**
	 * Testa a previsão do modelo usando uma imagem fornecida.
	 * @param modelo modelo treinado
	 * @param caminhoImagem caminho da imagem de teste, com extensão. 
	 * @param prob se verdadeiro, é mostrada a probabilidade prevista de cada dígito
	 * pelo modelo. Se falsa, mostra apenas o dígito previsto.
	 */
	static void testarPrevisao(Sequencial modelo, String caminhoImagem, boolean prob) {
		String extensao = ".jpg";
		Tensor amostra = new Tensor(f.carregarImagemCinza("./mnist/teste/" + caminhoImagem + extensao));
		amostra.unsqueeze(0);//2d -> 3d

		Tensor p = modelo.forward(amostra);
		double[] prev = p.paraArrayDouble();
		
		System.out.print("\nTestando: " + caminhoImagem + extensao);
		if (prob) {
			System.out.println();
			for (int i = 0; i < prev.length; i++) {
				System.out.println("Prob: " + i + ": " + (int)(prev[i]*100) + "%");
			}
		
		} else {
			System.out.print(" -> Prev: " + f.maiorIndice(prev));
		}

	}

	/**
	 * Testa os acertos do modelo usando os dados de teste do MNIST.
	 * @param modelo modelo treinado.
	 */
	static void testarAcertosMNIST(Sequencial modelo) {
		final String caminho = "./mnist/teste/";
		
		final int digitos = 10;
		final int amostras = 100;
		double media = 0;
		for (int d = 0; d < digitos; d++) {
			final int digito = d;

			Tensor[] imagens = new Tensor[amostras];

			int numThreads = Runtime.getRuntime().availableProcessors();
			if (numThreads > amostras) numThreads = amostras;
			try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
				for (int a = 0; a < amostras; a++) {
					final int amostra = a;
					exec.submit(() -> {
						String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
						Tensor img = new Tensor(f.carregarImagemCinza(caminhoImagem));
						img.unsqueeze(0);// 2d -> 3d
						imagens[amostra] = img;
					});
				}
			}

			double acertos = 0;
			Tensor[] prevs = modelo.forwards(imagens);
			for (Tensor t : prevs) {
				if (f.maiorIndice(t.paraArray()) == digito) acertos++;
			}

			double porcentagem = acertos / (double)amostras;
			System.out.println("Acertos " + d + " -> " + porcentagem + "%");
			media += porcentagem;
		}

		System.out.println("média acertos: " + String.format("%.2f", (media/digitos)*100) + "%");
	}

	/**
	 * Copia o valor de brilho do tensor e distribuir para os canais RGB.
	 * @param tensor tensor desejado (1 canal de cor).
	 * @return tensor com 3 canais de cor.
	 */
	static Tensor tensorCinzaParaRGB(Tensor tensor) {
		if (tensor.numDim() != 2) {
			throw new IllegalArgumentException("\nTensor deve ser 2D.");
		}

		int[] shape = tensor.shape();

		Tensor rgb = new Tensor(3, shape[0], shape[1]);

		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				double c = tensor.get(i, j);
				rgb.set(c, 0, i, j);
				rgb.set(c, 1, i, j);
				rgb.set(c, 2, i, j);
			}
		}

		return rgb;
	}

	/**
	 * Ajusta o nível de cor do tensor.
	 * <p>
	 * 		Valores devem estar no intervalo {@code [0, 1]}
	 * </p>
	 * @param t {@code Tensor} desejado.
	 * @param r intensidade de cor {@code VERMELHA} desejada.
	 * @param g intensidade de cor {@code VERDE} desejada.
	 * @param b intensidade de cor {@code AZUL} desejada.
	 */
	static void coresTensor(Tensor t, double r, double g, double b) {
		if (t.numDim() != 3) {
			throw new IllegalArgumentException(
				"\nTensor deve ser 3D (RGB)."
			);
		}

		int[] shape = t.shape();

		
		final double R = Math.min(1.0, Math.max(r, 0.0));
		final double G = Math.min(1.0, Math.max(g, 0.0));
		final double B = Math.min(1.0, Math.max(b, 0.0));

		t.slice(new int[]{0, 0, 0}, new int[]{1, shape[1], shape[2]}).aplicar(x -> x*R);//r
		t.slice(new int[]{1, 0, 0}, new int[]{2, shape[1], shape[2]}).aplicar(x -> x*G);//g
		t.slice(new int[]{2, 0, 0}, new int[]{3, shape[1], shape[2]}).aplicar(x -> x*B);//b
	}
}
