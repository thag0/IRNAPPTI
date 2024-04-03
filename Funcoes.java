import java.awt.image.BufferedImage;
import java.io.File;

import geim.Geim;
import geim.Pixel;
import render.Janela;
import rna.camadas.Convolucional;
import rna.core.Mat;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;

public class Funcoes{
   final String CAMINHO_IMAGEM;
   private Geim geim = new Geim();

   /**
    * Auxiliar contendo funções úteis para análise e testes.
    * @param caminhoImagem diretório contendo imagens do {@code MNIST}
    */
   public Funcoes(String caminhoImagem){
      this.CAMINHO_IMAGEM = caminhoImagem;
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
   public double entropiaCondicional(double[] previsoes){
      double ec = 0;
      for(double prev : previsoes){
         ec += prev * Math.log(prev);
      }
      return -ec;
   }

   /**
    * Região mais significativa para o modelo prever o dígito.
    * @param modelo modelo treinado.
    * @param entrada amostra de entrada.
    * @param rotulo rótulo correspondente à amostra.
    */
   public void gradCAM(Sequencial modelo, Tensor4D entrada, double[] rotulo){
      //passo de backpropagation para ter os gradientes calculados
      Tensor4D prev = modelo.forward(entrada);
      double[] derivadas = modelo.perda().derivada(prev.paraArray(), rotulo);
      Tensor4D grad = new Tensor4D(derivadas);
      int numCamadas = modelo.numCamadas();
      for(int i = numCamadas-1; i >= 0; i--){
         grad = modelo.camada(i).backward(grad);
      }

      //pegar índice da camada convolucional do modelo
      int idConv = 0;
      for(int i = 0; i < modelo.numCamadas(); i++){
         if(modelo.camada(i) instanceof Convolucional){
            idConv = i;
            break;
         }
      }

      Convolucional conv = (Convolucional) modelo.camada(idConv);
      
      //aplicar o grad cam
      Tensor4D convGrad = conv.gradSaida.clone();
      Tensor4D convSaida = conv.saida().clone();
      int canais = convGrad.dim2();
      int altura = convSaida.dim3(), largura = convSaida.dim4();
      Tensor4D mapa = new Tensor4D(altura, largura);

      for(int i = 0; i < canais; i++){
         Tensor4D tempGrad = convGrad.subTensor2D(0, i);
         Tensor4D tempSaida = convSaida.subTensor2D(0, i);
         tempGrad.mult(tempSaida);

         double peso = tempGrad.somar() / (altura * largura);
         Tensor4D t = convSaida.subTensor2D(0, i);
         t.map(x -> x * peso);
         
         mapa.add(t); 
      }

      //grad cam aplica a relu ao final do processo
      mapa.relu();

      //normalização
      double min = mapa.minimo(), max = mapa.maximo();
      mapa.map(x -> (x - min) / (max - min));
      
      //desenhar os valores calculados
      int escala = 15;
      desenharImagem(mapa, escala, false, "Mapa");
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
   public void desenharSaidas(Convolucional conv, Tensor4D amostra, int escala, boolean normalizar){
      Tensor4D prev = conv.forward(amostra);
      int filtros = conv.numFiltros();

      Tensor4D[] arr = new Tensor4D[filtros];
      for(int i = 0; i < arr.length; i++){
         arr[i] = prev.subTensor2D(0, i);
      }
      
      desenharImagens(arr, escala, normalizar, "Saidas Conv");
   }

   /**
    * Desenha o conteúdo 2d do tensor em uma janela gráfica.
    * @param tensor tensor com os dados desejados.
    * @param escala escala de ampliação da janela.
    * @param normalizar normalizar os valores do tensor entre 0 e 1
    * @param titulo nome da janela.
    */
   public void desenharImagem(Tensor4D tensor, int escala, boolean normalizar, String titulo){
      if(normalizar) normalizar(tensor);
      Janela janela = new Janela(tensor.dim3(), tensor.dim4(), escala, titulo);
      janela.desenharImagem(tensor);
   }

   /**
    * Desenha matriz por matriz dentro do array.
    * @param arr array de tensores.
    * @param escala escala de ampliação da janela.
    * @param normalizar normalizar os valores entre 0 e 1.
    */
   public void desenharImagens(Tensor4D[] arr, int escala, boolean normalizar, String titulo){
      int[] dim = {
         arr[0].dim3(), 
         arr[0].dim4()
      };

      Janela janela = new Janela(dim[0], dim[1], escala, titulo);

      if(normalizar){
         for(Tensor4D t : arr){
            normalizar(t);
         }
      }

      janela.desenharImagens(arr);
      
   }

   /**
    * Salva os resultados das ativações e pré ativações de uma camada 
    * convolucional do modelo
    * @param modelo modelo desejado.
    * @param idConv índice da camada convolucional do modelo.
    * @param normalizar normaliza os valores entre 0 e 1.
    */
   public void exportarAtivacoes(Sequencial modelo, int idConv, boolean normalizar, int escala){
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

         Tensor4D prev = modelo.forward(amostra);// ver as saídas calculadas

         Mat[] somatorios = new Mat[camada.somatorio.dim2()];
         Mat[] saidas = new Mat[prev.dim2()];
         for(int j = 0; j < saidas.length; j++){
            saidas[j] = new Mat(prev.array2D(0, j));
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
   public void exportarFiltros(Sequencial modelo, int idConv, boolean normalizar){
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
    * Salva os valores das matrizes como imagens no caminho especificado.
    * @param arr array de matrizes.
    * @param caminho diretório onde os arquivos serão salvos.
    */
   public void exportarMatrizes(Mat[] arr, int escala, String caminho){
      for(int i = 0; i < arr.length; i++){
         normalizar(arr[i]);
         exportarImagem(arr[i], (caminho + "amostra-" + (i+1)), escala);
      }
   }

   /**
    * 
    * @param mat
    * @param caminho
    * @param escala
    */
   public void exportarImagem(Mat mat, String caminho, double escala){
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

   /**
    * Normaliza os valores dentro da matriz.
    * @param mat matriz base.
    */
   public void normalizar(Mat mat){
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
    * Normaliza os valores dentro do tensor.
    * @param tensor matriz base.
    */
   public void normalizar(Tensor4D tensor){
      double min = tensor.minimo(); 
      double max = tensor.maximo();

      final double minimo = min, maximo = max;

      tensor.map((x) -> {
         return (x - minimo) / (maximo - minimo);
      });
   }

   /**
    * Converte uma imagem numa matriz contendo seus valores de brilho entre 0 e 1.
    * @param caminho caminho da imagem.
    * @return matriz contendo os valores de brilho da imagem.
    */
   public double[][] imagemParaMatriz(String caminho){
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
    * Limpa os arquivos do diretório.
    * @param caminho caminho do diretório.
    */
   public void limparDiretorio(String caminho){
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
    * Calcula o índice que contém o maior valor no array.
    * @param arr array base.
    * @return índice com o maior valor.
    */
   public int maiorIndice(double[] arr){
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
   public double[] gerarRotuloMnist(int digito){
      double[] arr = new double[10];

      for(int i = 0; i < arr.length; i++){
         arr[i] = 0.0;
      }
      arr[digito] = 1.0;

      return arr;
   }
}
