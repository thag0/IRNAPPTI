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

      String nomeModelo = "conv-mnist-95-7";
      Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELO + nomeModelo + ".txt");

      final int digito = 4;
      Tensor4D amostra = new Tensor4D(funcoes.imagemParaMatriz(CAMINHO_IMAGEM +  digito + "/img_3.jpg"));
      Tensor4D prev = modelo.forward(amostra);

      double[] rotulo = funcoes.gerarRotuloMnist(digito);
      funcoes.gradCAM(modelo, amostra, rotulo, true);

      prev.reformatar(10, 1);
      prev.print(8);
      System.out.println("Dígito " + digito + ", Previsto: " + funcoes.maiorIndice(prev.paraArray()));

      double ec = funcoes.entropiaCondicional(prev.paraArray());
      System.out.println("Entropia condicional: " + String.format("%.2f", (100 - (ec * 100))));

      // boolean normalizar = true;
      // funcoes.exportarAtivacoes(modelo, 0, normalizar, 20);
      // funcoes.exportarAtivacoes(modelo, 2, normalizar, 20);
      // funcoes.exportarFiltros(modelo, 0, normalizar);
      // funcoes.exportarFiltros(modelo, 2, normalizar);
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
      String extensao = ".jpg";
      Tensor4D amostra = new Tensor4D(funcoes.imagemParaMatriz("/mnist/teste/" + caminhoImagem + extensao));
      modelo.forward(amostra);
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
            
            modelo.forward(img);
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