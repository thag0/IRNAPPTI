import ged.Ged;
import geim.Geim;
import jnn.core.OpTensor4D;
import jnn.core.Tensor4D;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;

public class AnaliseModelo{
	static final String CAMINHO_MODELO = "./modelos/jnn/";
	static final String CAMINHO_IMAGEM = "./mnist/teste/";

	static Ged ged = new Ged();
	static Geim geim = new Geim();
	static OpTensor4D optensor = new OpTensor4D();
	static Funcional f = new Funcional();
	static Serializador serializador = new Serializador();
	
	public static void main(String[] args) {
		ged.limparConsole();

		// String nomeModelo = "modelo-treinado";
		String nomeModelo = "conv-mnist-95-1";
		// String nomeModelo = "conv-mnist-94-4";
		// String nomeModelo = "mlp-mnist-89";
		Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELO + nomeModelo + ".nn");

		final int digito = 3;
		Tensor4D amostra = f.carregarImagemCinza(CAMINHO_IMAGEM +  digito + "/img_2.jpg");
		
		double[] rotulo = f.gerarRotuloMnist(digito);
		Tensor4D heatmap = f.gradCAM(modelo, amostra, rotulo);
		Tensor4D heatpmapRGB = tensorCinzaParaRGB(heatmap);
		Tensor4D amostraRGB = tensorCinzaParaRGB(amostra);

		amostraRGB.map(x -> x*0.95);
		heatpmapRGB.map2D(0, 0, x -> x*0.6)// r
			.map2D(0, 1, x -> x*0.2)// g 
			.map2D(0, 2, x -> x*0.9);// b

		// f.desenharImagem(heatpmapRGB, 10, true, "Heatmap");
		// f.desenharImagem(amostraRGB, 10, false, "Amostra");
		// f.desenharImagem(amostraRGB.clone().add(heatpmapRGB), 10, false, "Heatmap + Amostra");
		
		// f.desenharMnist(modelo);

		f.matrizConfusao(modelo, 100);

		// f.desenharSaidas(modelo.camada(0), amostra, 20, true);

		// Tensor4D prev = modelo.forward(amostra);
		// prev.reshape(10, 1);
		// prev.print(10);
		// System.out.println("Dígito: " + digito + " -> Previsto: " + f.maiorIndice(prev.paraArray()));

		// double ec = f.entropiaCondicional(prev.paraArray());
		// System.out.println("Entropia condicional: " + String.format("%.2f", (100 - (ec * 100))));

		// new Thread(() -> {
		// 	boolean normalizar = true;
		// 	int escala = 20;
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
		Tensor4D amostra = f.carregarImagemCinza("./mnist/teste/" + caminhoImagem + extensao);

		double[] prev = modelo.forward(amostra).paraArray();
		
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
		String caminho = "./mnist/teste/";
		
		int digitos = 10;
		int amostras = 100;
		double media = 0;
		for (int digito = 0; digito < digitos; digito++) {

			double acertos = 0;
			for (int amostra = 0; amostra < amostras; amostra++) {
				String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
				Tensor4D img = f.carregarImagemCinza(caminhoImagem);
				
				modelo.forward(img);
				double[] previsoes = modelo.saidaParaArray();
				if (f.maiorIndice(previsoes) == digito) {
					acertos++;
				}
			}
			double porcentagem = acertos / (double)amostras;

			media += porcentagem;
			System.out.println("Acertos " + digito + " -> " + porcentagem);
		}
		System.out.println("média acertos: " + String.format("%.2f", (media/digitos)*100) + "%");
	}

	/**
	 * Copia o valor de brilho do tensor e distribuir para os canais RGB.
	 * @param tensor tensor desejado (1 canal de cor).
	 * @return tensor com 3 canais de cor.
	 */
	static Tensor4D tensorCinzaParaRGB(Tensor4D tensor) {
		double[][] cinza = tensor.array2D(0, 0);

		Tensor4D rgb = new Tensor4D(3, tensor.dim3(), tensor.dim4());

		rgb.copiar(cinza, 0, 0)
		   .copiar(cinza, 0, 1)
		   .copiar(cinza, 0, 2);

		return rgb;
	}

}