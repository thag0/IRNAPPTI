import java.awt.image.BufferedImage;
import java.io.File;

import ged.Ged;
import geim.Geim;
import geim.Pixel;
import render.Janela;
import rna.camadas.*;
import rna.core.Mat;
import rna.core.OpMatriz;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class AnaliseModelo{
   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static final String caminhoModelo = "./modelos/conv-mnist-89.txt";
   static final String caminhoImagem = "/mnist/teste/";
   
   public static void main(String[] args){
      ged.limparConsole();

      var modelo = new Serializador().lerSequencial("./modelos/modelo-convolucional.txt");
      modelo.info();

      // exportarAtivacoes(modelo, 0);
      // exportarAtivacoes(modelo, 2);
   }

   /**
    * Salva os resultados das ativações e pré ativações de uma camada 
    * convolucional do modelo
    * @param modelo modelo para análise.
    */
   static void exportarAtivacoes(Sequencial modelo, int idConv){
      Convolucional camada;
      try{
         camada = (Convolucional) modelo.camada(idConv);
      }catch(Exception e){
         System.out.println(
            "Ocorreu um erro ao tentar converter a camada com id recebido (" + idConv + ")."
         );
         throw new RuntimeException(e);
      }

      String diretorioCamada = "conv" + ((idConv == 0) ? "1" : "2");

      final int digitos = 10;
      for(int i = 0; i < digitos; i++){
         String caminhoAmostra = caminhoImagem + i + "/img_3.jpg";
         var imagem = imagemParaMatriz(caminhoAmostra);
         var amostra = new double[][][]{imagem};
         modelo.calcularSaida(amostra);// ver as saídas calculadas

         Mat[] somatorios = new Mat[camada.somatorio.length];
         Mat[] saidas = new Mat[camada.saida.length];
         for(int j = 0; j < saidas.length; j++){
            saidas[j] = camada.saida[j];
            somatorios[j] = camada.somatorio[j];
         }
   
         int escala = 20;
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
    */
   static void desenharMatriz(Mat mat, int escala){
      normalizar(mat);
      Janela janela = new Janela(mat.lin(), mat.col(), escala);
      janela.desenharMat(mat);
   }

   /**
    * Desenha matriz por matriz dentro do array.
    * @param arr array de matrizes.
    * @param escala escala de ampliação da janela.
    */
   static void desenharMatrizes(Mat[] arr, int escala){
      int[] dim = {
         arr[0].lin(), 
         arr[0].col()
      };
      Janela janela = new Janela(dim[0], dim[1], escala);

      for(Mat m : arr){
         normalizar(m);
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
            else if(valor > max) max = valor;
         }
      }

      final double minimo = min, maximo = max;

      mat.map((x) -> {
         return x = (x - minimo) / (maximo - minimo);
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