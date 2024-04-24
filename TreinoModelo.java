import java.awt.image.BufferedImage;
import java.text.DecimalFormat;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

import ged.Dados;
import ged.Ged;
import geim.Geim;
import jnn.camadas.Convolucional;
import jnn.camadas.Densa;
import jnn.camadas.Dropout;
import jnn.camadas.Entrada;
import jnn.camadas.Flatten;
import jnn.camadas.MaxPooling;
import jnn.core.Tensor4D;
import jnn.modelos.Modelo;
import jnn.modelos.Sequencial;
import jnn.serializacao.Serializador;

public class TreinoModelo{
   static Ged ged = new Ged();
   static Geim geim = new Geim();

   // dados de controle
   static final int NUM_DIGITOS_TREINO = 10;
   static final int NUM_DIGITOS_TESTE  = NUM_DIGITOS_TREINO;
   static final int NUM_AMOSTRAS_TREINO = 400;
   static final int NUM_AMOSTRAS_TESTE  = 100;
   static final int TREINO_EPOCAS = 12;
   static final int TREINO_LOTE = 1;
   static final boolean TREINO_LOGS = true;

   // caminhos de arquivos externos
   static final String CAMINHO_TREINO = "/dados/mnist/treino/";
   static final String CAMINHO_TESTE = "/dados/mnist/teste/";
   static final String CAMINHO_SAIDA_MODELO = "./dados/modelos/modelo-treinado.nn";
   static final String CAMINHO_HISTORICO = "historico-perda";

   public static void main(String[] args) {
      ged.limparConsole();
      
      final var treinoX = new Tensor4D(carregarDadosMNIST(CAMINHO_TREINO, NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO));
      final var treinoY = criarRotulosMNIST(NUM_AMOSTRAS_TREINO, NUM_DIGITOS_TREINO);

      Sequencial modelo = criarModelo();
      modelo.setHistorico(true);
      modelo.info();

      System.out.println("Treinando.");
      long tempo = System.nanoTime();
         modelo.treinar(treinoX, treinoY, TREINO_EPOCAS, TREINO_LOTE, TREINO_LOGS);
      tempo = System.nanoTime() - tempo;

      long segundosTotais = TimeUnit.NANOSECONDS.toSeconds(tempo);
      long horas = segundosTotais / 3600;
      long minutos = (segundosTotais % 3600) / 60;
      long segundos = segundosTotais % 60;

      System.out.println("\nTempo de treino: " + horas + "h " + minutos + "min " + segundos + "s");
      System.out.print("Treino -> perda: " + modelo.avaliar(treinoX, treinoY) + " - ");
      System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(treinoX, treinoY) * 100), 4) + "%");

      System.out.println("\nCarregando dados de teste.");
      final var testeX = carregarDadosMNIST(CAMINHO_TESTE, NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE);
      final var testeY = criarRotulosMNIST(NUM_AMOSTRAS_TESTE, NUM_DIGITOS_TESTE);
      System.out.print("Teste -> perda: " + modelo.avaliar(testeX, testeY) + " - ");
      System.out.println("acurácia: " + formatarDecimal((modelo.avaliador().acuracia(testeX, testeY) * 100), 4) + "%");

      exportarHistorico(modelo, CAMINHO_HISTORICO);
      salvarModelo(modelo, CAMINHO_SAIDA_MODELO);
   }

   /*
    * Criação de modelos para testes.
    */
    static Sequencial criarModelo(){
      Sequencial modelo = new Sequencial(
         new Entrada(28, 28),
         new Convolucional(new int[]{3, 3}, 20, "leaky-relu"),
         new MaxPooling(new int[]{2, 2}),
         new Convolucional(new int[]{3, 3}, 22, "leaky-relu"),
         new MaxPooling(new int[]{2, 2}),
         new Flatten(),
         new Densa(128, "sigmoid"),
         new Dropout(0.25),
         new Densa(NUM_DIGITOS_TREINO, "softmax")
      );

      modelo.compilar("sgd", "entropia-cruzada");
      return modelo;
   }

   /**
    * Salva o modelo num arquivo separado
    * @param modelo modelo desejado.
    * @param caminho caminho de destino
    */
   static void salvarModelo(Sequencial modelo, String caminho){
      System.out.println("Salvando modelo.");
      new Serializador().salvar(modelo, caminho, "double");
   }

   /**
    * Converte uma imagem numa matriz contendo seus valores de brilho entre 0 e 1.
    * @param caminho caminho da imagem.
    * @return matriz contendo os valores de brilho da imagem.
    */
   static double[][] imagemParaMatriz(String caminho){
      BufferedImage img = geim.lerImagem(caminho);
      double[][] imagem = new double[img.getHeight()][img.getWidth()];

      int[][] cinza = geim.obterCinza(img);

      for(int y = 0; y < imagem.length; y++){
         for(int x = 0; x < imagem[y].length; x++){
            imagem[y][x] = (double)cinza[y][x] / 255;
         }
      }
      return imagem;
   }

   /**
    * Testa as previsões do modelo no formato de probabilidade.
    * @param modelo modelo sequencial de camadas.
    * @param imagemTeste nome da imagem que deve estar no diretório /minst/teste/
    */
   static void testarPorbabilidade(Sequencial modelo, String imagemTeste){
      System.out.println("\nTestando: " + imagemTeste);
      double[][][] teste1 = new double[1][][];
      teste1[0] = imagemParaMatriz("/dados/mnist/teste/" + imagemTeste + ".jpg");
      modelo.forward(teste1);
      double[] previsao = modelo.saidaParaArray();
      for(int i = 0; i < previsao.length; i++){
         System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
      }
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
      final int numThreads = Runtime.getRuntime().availableProcessors()/2;

      try (ExecutorService exec = Executors.newFixedThreadPool(numThreads)) {
         int id = 0;
         for (int i = 0; i < digitos; i++) {
            for (int j = 0; j < amostras; j++) {
               final String caminhoCompleto = caminho + i + "/img_" + j + ".jpg";
               final int indice = id;
               exec.submit(() -> {
                  double[][] imagem = imagemParaMatriz(caminhoCompleto);
                  imagens[indice][0] = imagem;
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
   static double[][] criarRotulosMNIST(int amostras, int digitos){
      double[][] rotulos = new double[digitos * amostras][digitos];
      for(int numero = 0; numero < digitos; numero++){
         for(int i = 0; i < amostras; i++){
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
   static String formatarDecimal(double valor, int casas){
      String valorFormatado = "";

      String formato = "#.";
      for(int i = 0; i < casas; i++) formato += "#";

      DecimalFormat df = new DecimalFormat(formato);
      valorFormatado = df.format(valor);

      return valorFormatado;
   }

   /**
    * Salva um arquivo csv com o historico de desempenho do modelo.
    * @param modelo modelo.
    * @param caminho caminho onde será salvo o arquivo.
    */
   static void exportarHistorico(Modelo modelo, String caminho){
      System.out.println("Exportando histórico de perda");
      double[] perdas = modelo.historico();
      double[][] dadosPerdas = new double[perdas.length][1];

      for(int i = 0; i < dadosPerdas.length; i++){
         dadosPerdas[i][0] = perdas[i];
      }

      Dados dados = new Dados(dadosPerdas);
      ged.exportarCsv(dados, caminho);
   }
}
