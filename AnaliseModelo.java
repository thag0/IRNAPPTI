import java.awt.image.BufferedImage;
import java.io.File;

import ged.Ged;
import geim.Geim;
import geim.Pixel;
import render.Janela;
import rna.camadas.*;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class AnaliseModelo{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();

   static final String CAMINHO_MODELO = "./modelos/rna/";
   static final String CAMINHO_IMAGEM = "/mnist/teste/";
   
   public static void main(String[] args){
      ged.limparConsole();

      String nomeModelo = "conv-mnist-93";
      Sequencial modelo = new Serializador().lerSequencial(CAMINHO_MODELO + nomeModelo + ".txt");

      int digito = 1;
      // Tensor4D amostra = new Tensor4D(imagemParaMatriz(CAMINHO_IMAGEM +  digito + "/img_0.jpg"));
      Tensor4D amostra = new Tensor4D(imagemParaMatriz("/mnist/3_deslocado.jpg"));
      modelo.calcularSaida(amostra);

      gradCAM(modelo, amostra, gerarRotuloMnist(digito));

      Tensor4D saida = modelo.camadaSaida().saida();
      saida.reformatar(10, 1);
      saida.print(4);
      System.out.println("Previsto: " + maiorIndice(saida.paraArray()));
      System.out.println("Entropia condicional: " + (1- entropiaCondicional(modelo.saidaParaArray())));

      // boolean normalizar = true;
      // exportarAtivacoes(modelo, 0, normalizar, 20);
      // exportarAtivacoes(modelo, 2, normalizar, 20);
      // exportarFiltros(modelo, 0, normalizar);
      // exportarFiltros(modelo, 2, normalizar);
   }

   /**
    * Região mais significativa para a rede definir o dígito.
    * @param modelo modelo treinado.
    * @param entrada amostra de entrada.
    * @param rotulo rótulo correspondente à amostra.
    */
   static void gradCAM(Sequencial modelo, Tensor4D entrada, double[] rotulo){
      //passo de backpropagation
      modelo.calcularSaida(entrada);
      double[] derivadas = modelo.perda().derivada(modelo.saidaParaArray(), rotulo);
      int numCamadas = modelo.numCamadas();
      modelo.camada(numCamadas-1).calcularGradiente(new Tensor4D(derivadas));
      for(int i = numCamadas-2; i >= 0; i--){
         modelo.camada(i).calcularGradiente(modelo.camada(i+1).gradEntrada());
      }

      //passo de análise
      Convolucional conv = (Convolucional) modelo.camada(2);
      Tensor4D gradiente = conv.gradSaida.clone();

      Tensor4D mapa = new Tensor4D(gradiente.dim3(), gradiente.dim4());
      mapa.nome("mapa");
      for(int i = 0 ; i < gradiente.dim2(); i++){
         Tensor4D temp = new Tensor4D(gradiente.array2D(0, i));
         double media = temp.somar() / temp.tamanho();

         Tensor4D saidas = conv.saida().clone();
         saidas.map(x -> x*media);

         for(int j = 0; j < saidas.dim2(); j++){
            mapa.add(saidas.subTensor2D(0, j));
         }
      }

      desenharMatriz(new Mat(mapa.array2D(0, 0)), 20, true);
      
      for(int i = 0; i < conv.entrada.dim2(); i++){
         mapa.add(conv.saida.subTensor2D(0, i));
      }
      mapa.print(5);
      desenharMatriz(new Mat(mapa.array2D(0, 0)), 20, true);
   }

   static double[] gerarRotuloMnist(int digito){
      double[] arr = new double[10];
      for(int i = 0; i < arr.length; i++){
         arr[i] = 0;
      }
      arr[digito] = 1;

      return arr;
   }

   /**
    * Calcula o valor de incerteza do modelo em relação as sua previsões.
    * <p>
    *    Valores mais baixos indicam menor incerteza do modelo, que significa
    *    que o modelo tem bastante "confiança" na previsão feita.
    * </p>
    * @param previsoes previsões do modelo.
    * @return valor de entropia condicional com base nas previsões.
    */
   static double entropiaCondicional(double[] previsoes){
      double ec = 0;
      for(double prev : previsoes){
         ec += prev * Math.log(prev);
      }
      return -ec;
   }

   /**
    * Abre uma janela gráfica contendo a saída da camada fornecida.
    * <p>
    *    É necessário que a camada tenha pré calculado algum resultado para
    *    que ele poda ser visualizado.
    * </p>
    * @param conv camada convolucional.
    * @param escala escala de ampliação da imagem original.
    * @param normalizar normaliza os valores entre 0 e 1 para evitar artefatos
    * na janela gráfica.
    */
   static void desenharSaidas(Convolucional conv, int escala, boolean normalizar){
      Mat[] arr = new Mat[conv.numFiltros()];
      for(int i = 0; i < arr.length; i++){
         arr[i] = new Mat(conv.saida.array2D(0, i));
      }
      
      desenharMatrizes(arr, escala, normalizar);
   }

   /**
    * Usa o modelo para prever todos os dados de teste.
    * @param modelo modelo treinado.
    */
   static void testarTodosDados(Sequencial modelo){
      final int digitos = 10;
      final int amostras = 100;
      for(int i = 0; i < digitos; i++){
         for(int j = 0; j < amostras; j++){
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
   static void testarPrevisao(Sequencial modelo, String caminhoImagem, boolean prob){
      double[][][] entrada = new double[1][][];
      String extensao = ".jpg";
      entrada[0] = imagemParaMatriz("/mnist/teste/" + caminhoImagem + extensao);
      modelo.calcularSaida(entrada);
      double[] previsao = modelo.saidaParaArray();
      
      System.out.print("\nTestando: " + caminhoImagem + extensao);
      if(prob){
         System.out.println();
         for(int i = 0; i < previsao.length; i++){
            System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         }
      }else{
         System.out.print(" -> Prev: " + maiorIndice(previsao));
      }

   }

   /**
    * Testa os acertos do modelo usando os dados de teste do MNIST.
    * @param modelo modelo treinado.
    */
   static void testarAcertosMNIST(Sequencial modelo){
      String caminho = "/mnist/teste/";
      
      int digitos = 10;
      int amostras = 100;
      double media = 0;
      for(int digito = 0; digito < digitos; digito++){
         double acertos = 0;
         for(int amostra = 0; amostra < amostras; amostra++){
            String caminhoImagem = caminho + digito + "/img_" + amostra + ".jpg";
            Tensor4D img = new Tensor4D(imagemParaMatriz(caminhoImagem));
            
            modelo.calcularSaida(img);
            double[] previsoes = modelo.saidaParaArray();
            if(maiorIndice(previsoes) == digito){
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
    * Calcula o índice que contém o maior valor no array.
    * @param arr array base.
    * @return índice com o maior valor.
    */
   static int maiorIndice(double[] arr){
      int id = 0;
      double maior = arr[0];

      for(int i = 1; i < arr.length; i++){
         if(arr[i] > maior){
            id = i;
            maior = arr[i];
         }
      }

      return id;
   }

   /**
    * Salva os resultados das ativações e pré ativações de uma camada 
    * convolucional do modelo
    * @param modelo modelo desejado.
    * @param idConv índice da camada convolucional do modelo.
    * @param normalizar normaliza os valores entre 0 e 1.
    */
   static void exportarAtivacoes(Sequencial modelo, int idConv, boolean normalizar, int escala){
      Convolucional camada;
      try{
         camada = (Convolucional) modelo.camada(idConv);
      }catch(Exception e){
         throw new IllegalArgumentException(
            "\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
            modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
         );
      }

      String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");

      final int digitos = 10;
      for(int i = 0; i < digitos; i++){
         String caminhoAmostra = CAMINHO_IMAGEM + i + "/img_16.jpg";
         var imagem = imagemParaMatriz(caminhoAmostra);
         var amostra = new double[][][]{imagem};

         modelo.calcularSaida(amostra);// ver as saídas calculadas

         Mat[] somatorios = new Mat[camada.somatorio.dim2()];
         Mat[] saidas = new Mat[camada.saida.dim2()];
         for(int j = 0; j < saidas.length; j++){
            saidas[j] = new Mat(camada.saida.array2D(0, j));
            somatorios[j] = new Mat(camada.somatorio.array2D(0, j));

            if(normalizar){
               normalizar(saidas[j]);
               normalizar(somatorios[j]);
            }
         }

         String caminhoSomatorio = "./resultados/pre-ativacoes/" + diretorioCamada + "/" + i + "/";
         String caminhoSaida = "./resultados/ativacoes/" + diretorioCamada + "/" + i + "/";

         limparDiretorio(caminhoSomatorio);
         limparDiretorio(caminhoSaida);

         exportarMatrizes(somatorios, escala, caminhoSomatorio);
         exportarMatrizes(saidas, escala, caminhoSaida);
      }

      System.out.println("Ativações exportadas para a camada " + idConv);
   }

   /**
    * Exporta os filtros da camada convolucional
    * @param modelo modelo desejado.
    * @param idConv índice da camada convolucional do modelo.
    * @param normalizar normaliza os valores entre 0 e 1.
    */
   static void exportarFiltros(Sequencial modelo, int idConv, boolean normalizar){
      Convolucional camada;
      try{
         camada = (Convolucional) modelo.camada(idConv);
      }catch(Exception e){
         throw new IllegalArgumentException(
            "\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
            modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
         );
      }

      String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");
      String caminho = "./resultados/filtros/" + diretorioCamada + "/";

      Tensor4D filtros = camada.filtros;
      limparDiretorio(caminho);

      int numFiltros = filtros.dim1();
      Mat[] arrFiltros = new Mat[numFiltros];
      for(int i = 0; i < numFiltros; i++){
         arrFiltros[i] = new Mat(filtros.array2D(i, 0));
         if(normalizar) normalizar(arrFiltros[i]);
      }

      exportarMatrizes(arrFiltros, 20, caminho);

      System.out.println("Filtros exportados para a camada " + idConv);
   }

   /**
    * Limpa os arquivos do diretório.
    * @param caminho caminho do diretório.
    */
   static void limparDiretorio(String caminho){
      File diretorio = new File(caminho);
  
      if(diretorio.isDirectory()){
         File[] arquivos = diretorio.listFiles();
  
         if(arquivos != null){
            for (File arquivo : arquivos){
               if(arquivo.isFile()){
                  arquivo.delete();
               }
            }
         }

      }else{
         System.out.println("O caminho fornecido não é um diretório válido.");
      }
   }

   /**
    * Salva os valores das matrizes como imagens no caminho especificado.
    * @param arr array de matrizes.
    * @param caminho diretório onde os arquivos serão salvos.
    */
   static void exportarMatrizes(Mat[] arr, int escala, String caminho){
      for(int i = 0; i < arr.length; i++){
         normalizar(arr[i]);
         exportarImagem(arr[i], (caminho + "amostra-" + (i+1)), escala);
      }
   }

   /**
    * 
    * @param mat
    * @param escala
    * @param normalizar
    */
   static void desenharMatriz(Mat mat, int escala, boolean normalizar){
      if(normalizar) normalizar(mat);
      Janela janela = new Janela(mat.lin(), mat.col(), escala);
      janela.desenharMat(mat);
   }

   /**
    * Desenha matriz por matriz dentro do array.
    * @param arr array de matrizes.
    * @param escala escala de ampliação da janela.
    * @param normalizar normalizar os valores entre 0 e 1.
    */
   static void desenharMatrizes(Mat[] arr, int escala, boolean normalizar){
      int[] dim = {
         arr[0].lin(), 
         arr[0].col()
      };
      Janela janela = new Janela(dim[0], dim[1], escala);

      if(normalizar){
         for(Mat m : arr){
            normalizar(m);
         }
      }

      janela.desenharArray(arr);
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
    * Normaliza os valores dentro da matriz.
    * @param mat matriz base.
    */
   static void normalizar(Mat mat){
      int linhas = mat.lin();
      int colunas = mat.col();
      double min = mat.elemento(0, 0); 
      double max = mat.elemento(0, 0);

      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            double valor = mat.elemento(i, j);
            if(valor < min) min = valor;
            if(valor > max) max = valor;
         }
      }

      final double minimo = min, maximo = max;

      mat.map((x) -> {
         return (x - minimo) / (maximo - minimo);
      });
   }

   /**
    * 
    * @param mat
    * @param caminho
    * @param escala
    */
   static void exportarImagem(Mat mat, String caminho, double escala){
      int altura =  (int) (mat.lin() * escala);
      int largura = (int) (mat.col() * escala);
      Pixel[][] estrutura = new Pixel[altura][largura];
  
      for(int y = 0; y < altura; y++){
         for(int x = 0; x < largura; x++){
            int originalY = (int) (y / escala);
            int originalX = (int) (x / escala);

            double cinza = mat.elemento(originalY, originalX);
            cinza *= 255;
            estrutura[y][x] = new Pixel((int) cinza, (int) cinza, (int) cinza);
         }
      }
  
      geim.exportarImagemPng(estrutura, caminho);
   }
}