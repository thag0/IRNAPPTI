import ged.Ged;
import geim.Geim;
import rna.core.OpMatriz;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class AnaliseModelo{
   static final String CAMINHO_MODELO = "./modelos/rna/";
   static final String CAMINHO_IMAGEM = "/mnist/teste/";

   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpMatriz opmat = new OpMatriz();
   static OpTensor4D optensor = new OpTensor4D();
   static Funcoes funcoes = new Funcoes(CAMINHO_IMAGEM);
   static Serializador serializador = new Serializador();
   
   public static void main(String[] args){
      ged.limparConsole();

      String nomeModelo = "conv-mnist-94";
      Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELO + nomeModelo + ".txt");

      final int digito = 9;
      Tensor4D amostra = new Tensor4D(funcoes.imagemParaMatriz(CAMINHO_IMAGEM +  digito + "/img_0.jpg"));
      Tensor4D previsao = modelo.calcularSaida(amostra);

      funcoes.gradCAM(modelo, amostra, funcoes.gerarRotuloMnist(digito), true);

      previsao.reformatar(10, 1);
      previsao.print(8);
      System.out.println("Previsto: " + funcoes.maiorIndice(previsao.paraArray()));

      double ec = funcoes.entropiaCondicional(previsao.paraArray());
      ec *= 100;//visualizar em %
      System.out.println("Entropia condicional: " + String.format("%.2f", (100 - ec)));

      // boolean normalizar = true;
      // exportarAtivacoes(modelo, 0, normalizar, 20);
      // exportarAtivacoes(modelo, 2, normalizar, 20);
      // exportarFiltros(modelo, 0, normalizar);
      // exportarFiltros(modelo, 2, normalizar);
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
      entrada[0] = funcoes.imagemParaMatriz("/mnist/teste/" + caminhoImagem + extensao);
      modelo.calcularSaida(entrada);
      double[] previsao = modelo.saidaParaArray();
      
      System.out.print("\nTestando: " + caminhoImagem + extensao);
      if(prob){
         System.out.println();
         for(int i = 0; i < previsao.length; i++){
            System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         }
      }else{
         System.out.print(" -> Prev: " + funcoes.maiorIndice(previsao));
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
            Tensor4D img = new Tensor4D(funcoes.imagemParaMatriz(caminhoImagem));
            
            modelo.calcularSaida(img);
            double[] previsoes = modelo.saidaParaArray();
            if(funcoes.maiorIndice(previsoes) == digito){
               acertos++;
            }
         }
         double porcentagem = acertos / (double)amostras;
         media += porcentagem;
         System.out.println("Acertos " + digito + " -> " + porcentagem);
      }
      System.out.println("média acertos: " + String.format("%.2f", (media/digitos)*100) + "%");
   }

}