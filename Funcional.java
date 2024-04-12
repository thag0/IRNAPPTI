import java.awt.image.BufferedImage;
import java.io.File;

import ged.Dados;
import geim.Geim;
import geim.Pixel;
import render.Janela;
import render.realtime.JanelaDesenho;
import rna.camadas.Convolucional;
import rna.core.Mat;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;

@SuppressWarnings("deprecation")//para classe Mat
public class Funcional{

   /**
    * Caminho das imagens de teste do mnist.
    */
   static final String CAMINHO_IMAGEM = "/mnist/teste/";

   /**
    * Gerenciador de imagens.
    */
   private Geim geim = new Geim();

   /**
    * Interface funcional.
    */
   public Funcional(){}

   /**
    * Calcula o valor de incerteza do modelo em relação as sua previsões.
    * <p>
    *    Valores mais baixos indicam menor incerteza do modelo, que significa
    *    que o modelo tem bastante "confiança" na previsão feita.
    * </p>
    * @param previsoes previsões do modelo.
    * @return valor de entropia condicional com base nas previsões.
    */
   public double entropiaCondicional(double[] previsoes) {
      double ec = 0;
      for (double prev : previsoes) {
         ec += prev * Math.log(prev);
      }
      return -ec;
   }

   /**
    * Calcula a região mais significativa para o modelo fazer sua previsão.
    * @param modelo modelo treinado.
    * @param entrada amostra de entrada.
    * @param rotulo rótulo correspondente à amostra.
    * @return {@code Tensor} contendo o mapa de calor calculado.
    */
   public Tensor4D gradCAM(Sequencial modelo, Tensor4D entrada, double[] rotulo) {
      //passo de backpropagation para ter os gradientes calculados
      Tensor4D prev = modelo.forward(entrada);
      double[] derivadas = modelo.perda().derivada(prev.paraArray(), rotulo);
      Tensor4D grad = new Tensor4D(derivadas);
      for (int i = modelo.numCamadas()-1; i >= 0; i--) {
         grad = modelo.camada(i).backward(grad);
      }

      //pegar índice da última camada convolucional do modelo
      int idConv = -1;
      for (int i = 0; i < modelo.numCamadas(); i++) {
         if (modelo.camada(i) instanceof Convolucional) {
            idConv = i;
         }
      }

      if (idConv == -1) {
         throw new IllegalArgumentException(
            "\nNenhuma camada convolucional encontrada no modelo."
         );
      }

      Convolucional conv = (Convolucional) modelo.camada(idConv);
      
      //calcular mapa de calor
      Tensor4D convAtv = conv._saida.clone();
      Tensor4D convGrad = conv._gradSaida.clone();
      int canais  = convGrad.dim2();
      int altura  = convGrad.dim3();
      int largura = convGrad.dim4();
   
      Tensor4D heatmap = new Tensor4D(altura, largura);

      for (int c = 0; c < canais; c++) {
         double media = convGrad.subTensor2D(0, c).media();
         heatmap.add(
            convAtv.subTensor2D(0, c)
            .map(x -> x*media)
         );
      } 

      heatmap.relu().normalizar(0, 1);
      // heatmap.normalizar(0, 1);

      double[][] m = heatmap.array2D(0, 0);
      m = ampliarMatriz(m, 28, 28);
      heatmap = new Tensor4D(m);

      return heatmap;
   }

   /**
    * Desenha uma janela gráfica para testar o modelo treinado com o dataset
    * {@code MNIST} em tempo real.
    * @param modelo modelo treinado.
    */
   public void desenharMnist(Sequencial modelo) {
      final byte fator = 28;
      final int escala = 18;

      JanelaDesenho jd = new JanelaDesenho(fator*escala, fator*(escala*2), modelo);

      while (jd.isVisible()) {
         jd.atualizar();

         try {
            Thread.sleep(80);
         } catch (Exception e) {}
      }

      jd.dispose();
   }

   /**
    * Calcula a matriz de confusão das predições do modelo.
    * @param modelo modelo treinado.
    */
   public void matrizConfusao(Sequencial modelo) {
      int amostras = 100;
      int digitos = 10;
      Tensor4D entradas = new Tensor4D(carregarDadosMNIST(CAMINHO_IMAGEM, amostras, digitos));
      double[][] rotulos = criarRotulosMNIST(amostras, digitos);
      
      int[][] matrizConfusao = modelo.avaliador().matrizConfusao(entradas, rotulos);
      
      Dados matriz = new Dados(matrizConfusao);
      matriz.print();
   }

   /**
    * Carrega os dados de entrada do MNIST (apenas features).
    * @param amostras quantidade de amostras por dígito
    * @param digitos quantidade de dígitos, iniciando do dígito 0.
    * @return dados carregados.
    */
   double[][][][] carregarDadosMNIST(String caminho, int amostras, int digitos) {
      double[][][][] entradas = new double[digitos * amostras][1][][];

      int id = 0;
      for (int dig = 0; dig < digitos; dig++) {
         for (int ams = 0; ams < amostras; ams++) {
            String caminhoCompleto = caminho + dig + "/img_" + ams + ".jpg";
            double[][] imagem = carregarImagemCinza(caminhoCompleto).array2D(0, 0);
            entradas[id++][0] = imagem;
         }
      }

      return entradas;
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
   public void desenharSaidas(Convolucional conv, Tensor4D amostra, int escala, boolean norm) {
      Tensor4D prev = conv.forward(amostra);
      int filtros = conv.numFiltros();
      Tensor4D[] arr = new Tensor4D[filtros];

      for (int i = 0; i < arr.length; i++) {
         arr[i] = prev.subTensor2D(0, i);
      }
      
      desenharImagens(arr, escala, norm, "Saidas Conv");
   }

   /**
    * Desenha o conteúdo 2d do tensor em uma janela gráfica.
    * @param tensor tensor com os dados desejados.
    * @param escala escala de ampliação da janela.
    * @param norm normalizar os valores do tensor entre 0 e 1
    * @param titulo nome da janela.
    */
   public void desenharImagem(Tensor4D tensor, int escala, boolean norm, String titulo) {
      if (norm) tensor.normalizar(0, 1);
      Janela janela = new Janela(tensor.dim3(), tensor.dim4(), escala, titulo);
      janela.desenharImagem(tensor);
   }

   /**
    * Desenha matriz por matriz dentro do array.
    * @param arr array de tensores.
    * @param escala escala de ampliação da janela.
    * @param norm normalizar os valores entre 0 e 1.
    */
   public void desenharImagens(Tensor4D[] arr, int escala, boolean norm, String titulo) {
      int[] dim = {
         arr[0].dim3(), 
         arr[0].dim4()
      };

      Janela janela = new Janela(dim[0], dim[1], escala, titulo);

      if (norm) {
         for (Tensor4D t : arr) {
            t.normalizar(0, 1);
         }
      }

      janela.desenharImagens(arr);
      
   }

   /**
    * Salva os resultados das ativações e pré ativações de uma camada 
    * convolucional do modelo
    * @param modelo modelo desejado.
    * @param idConv índice da camada convolucional do modelo.
    * @param norm normaliza os valores entre 0 e 1.
    */
   public void exportarAtivacoes(Sequencial modelo, int idConv, boolean norm, int escala) {
      Convolucional camada;
      try {
         camada = (Convolucional) modelo.camada(idConv);
      } catch (Exception e) {
         throw new IllegalArgumentException(
            "\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
            modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
         );
      }

      String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");

      final int digitos = 10;
      for (int i = 0; i < digitos; i++) {
         String caminhoAmostra = CAMINHO_IMAGEM + i + "/img_16.jpg";
         var amostra = carregarImagemCinza(caminhoAmostra);

         Tensor4D prev = modelo.forward(amostra);

         Mat[] somatorios = new Mat[camada._somatorio.dim2()];
         Mat[] saidas = new Mat[prev.dim2()];

         for (int j = 0; j < saidas.length; j++) {
            saidas[j] = new Mat(prev.array2D(0, j));
            somatorios[j] = new Mat(camada._somatorio.array2D(0, j));

            if (norm) {
               normalizar(saidas[j]);
               normalizar(somatorios[j]);
            }
         }

         String caminhoSomatorio = "./resultados/pre-ativacoes/" + diretorioCamada + "/" + i + "/";
         String caminhoSaida = "./resultados/ativacoes/" + diretorioCamada + "/" + i + "/";

         limparDiretorio(caminhoSomatorio);
         limparDiretorio(caminhoSaida);

         exportarMatrizes(saidas, escala, caminhoSaida);
         exportarMatrizes(somatorios, escala, caminhoSomatorio);
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
      Convolucional camada;
      try {
         camada = (Convolucional) modelo.camada(idConv);
      } catch (Exception e) {
         throw new IllegalArgumentException(
            "\nCamada com id " + idConv + " não é do tipo Convolucional e sim " + 
            modelo.camada(idConv).getClass().getSimpleName() + ", escolha um id válido."
         );
      }

      String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");
      String caminho = "./resultados/filtros/" + diretorioCamada + "/";

      Tensor4D filtros = camada._filtros;
      limparDiretorio(caminho);

      int numFiltros = filtros.dim1();
      Mat[] arrFiltros = new Mat[numFiltros];
      for (int i = 0; i < numFiltros; i++) {
         arrFiltros[i] = new Mat(filtros.array2D(i, 0));
         if (norm) normalizar(arrFiltros[i]);
      }

      exportarMatrizes(arrFiltros, escala, caminho);

      System.out.println("Filtros exportados para a camada " + idConv);
   }

   /**
    * Salva os valores das matrizes como imagens no caminho especificado.
    * @param arr array de matrizes.
    * @param caminho diretório onde os arquivos serão salvos.
    */
   public void exportarMatrizes(Mat[] arr, int escala, String caminho) {
      for (int i = 0; i < arr.length; i++) {
         normalizar(arr[i]);
         exportarImagem(arr[i], (caminho + "amostra-" + (i+1)), escala);
      }
   }

   /**
    * Salva a matriz num arquivo de imagem externo.
    * @param mat matriz desejada.
    * @param caminho diretório de destino.
    * @param escala escala de tratamento da imagem final.
    */
   public void exportarImagem(Mat mat, String caminho, double escala) {
      int altura =  (int) (mat.lin() * escala);
      int largura = (int) (mat.col() * escala);
      Pixel[][] estrutura = new Pixel[altura][largura];
  
      for (int y = 0; y < altura; y++) {
         for (int x = 0; x < largura; x++) {
            int originalY = (int) (y / escala);
            int originalX = (int) (x / escala);

            double cinza = mat.elemento(originalY, originalX);
            cinza *= 255;
            estrutura[y][x] = new Pixel((int) cinza, (int) cinza, (int) cinza);
         }
      }

      File diretorio = new File(caminho).getParentFile();
      if (!diretorio.exists()) diretorio.mkdirs();
  
      geim.exportarPng(estrutura, caminho);
   }

   /**
    * Normaliza os valores dentro da matriz.
    * @param mat matriz base.
    */
   public void normalizar(Mat mat) {
      int linhas = mat.lin();
      int colunas = mat.col();
      double min = mat.elemento(0, 0); 
      double max = mat.elemento(0, 0);

      for (int i = 0; i < linhas; i++) {
         for (int j = 0; j < colunas; j++) {
            double valor = mat.elemento(i, j);
            if (valor < min) min = valor;
            if (valor > max) max = valor;
         }
      }

      final double minimo = min, maximo = max;

      mat.map((x) -> {
         return (x - minimo) / (maximo - minimo);
      });
   }

   /**
    * Carrega a imagem a partir de um arquivo.
    * @param caminho caminho da imagem.
    * @return {@code Tensor} contendo os dados da imagem no
    * padrão RGB.
    */
   public Tensor4D carregarImagemRGB(String caminho) {
      BufferedImage img = geim.lerImagem(caminho);
      int altura = img.getHeight(), largura = img.getHeight();

      Tensor4D imagem = new Tensor4D(3, altura, largura);

      int[][] r = geim.obterVermelho(img);
      int[][] g = geim.obterVerde(img);
      int[][] b = geim.obterAzul(img);

      for (int y = 0; y < altura; y++) {
         for (int x = 0; x < largura; x++) {
            imagem.set((double)(r[y][x]) / 255, 0, 0, y, x);
            imagem.set((double)(g[y][x]) / 255, 0, 1, y, x);
            imagem.set((double)(b[y][x]) / 255, 0, 2, y, x);
         }
      }

      return imagem;
   }

   /**
    * Carrega a imagem a partir de um arquivo.
    * @param caminho caminho da imagem.
    * @return {@code Tensor} contendo os dados da imagem em
    * escala de cinza.
    */
   public Tensor4D carregarImagemCinza(String caminho) {
      BufferedImage img = geim.lerImagem(caminho);
      int altura = img.getHeight(), largura = img.getHeight();
      Tensor4D imagem = new Tensor4D(altura, largura);

      int[][] cinza = geim.obterCinza(img);   

      for (int y = 0; y < altura; y++) {
         for (int x = 0; x < largura; x++) {
            double c = (double)(cinza[y][x]) / 255;
            imagem.set(c, 0, 0, y, x);
         }  
      }

      return imagem;
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
      arr[digito] = 1.0;

      return arr;
   }

   public double[][] ampliarMatriz(double[][] m, int newWidth, int newHeight) {
      int height = m.length;
      int width = m[0].length;
      
      double[][] scaledMatrix = new double[newHeight][newWidth];
      
      for (int i = 0; i < newHeight; i++) {
         for (int j = 0; j < newWidth; j++) {
            double scaledHeight = (double) i / (newHeight - 1) * (height - 1);
            double scaledWidth = (double) j / (newWidth - 1) * (width - 1);
            
            int y0 = (int) scaledHeight;
            int x0 = (int) scaledWidth;
            int y1 = Math.min(y0 + 1, height - 1);
            int x1 = Math.min(x0 + 1, width - 1);
            
            double dx = scaledWidth - x0;
            double dy = scaledHeight - y0;
            
            double interpolatedValue = 
               (1 - dx) * (1 - dy) * m[y0][x0] +
               dx * (1 - dy) * m[y0][x1] +
               (1 - dx) * dy * m[y1][x0] +
               dx * dy * m[y1][x1];
            
            scaledMatrix[i][j] = interpolatedValue;
         }
      }
      
      return scaledMatrix;
   }
}
