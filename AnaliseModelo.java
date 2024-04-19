import ged.Ged;
import geim.Geim;
import rna.core.OpTensor4D;
import rna.core.Tensor4D;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class AnaliseModelo{
   static final String CAMINHO_MODELO = "./modelos/rna/";
   static final String CAMINHO_IMAGEM = "/mnist/teste/";

   static Ged ged = new Ged();
   static Geim geim = new Geim();
   static OpTensor4D optensor = new OpTensor4D();
   static Funcional f = new Funcional();
   static Serializador serializador = new Serializador();
   
   public static void main(String[] args){
      ged.limparConsole();

      String nomeModelo = "conv-mnist-95-6";
      // String nomeModelo = "conv-mnist-96-5";
      // String nomeModelo = "mlp-mnist-89";
      Sequencial modelo = serializador.lerSequencial(CAMINHO_MODELO + nomeModelo + ".nn");

      // f.matrizConfusao(modelo, 100);
      
      final int digito = 8;
      Tensor4D amostra = f.carregarImagemCinza(CAMINHO_IMAGEM +  digito + "/img_0.jpg");
      double[] rotulo = f.gerarRotuloMnist(digito);
      Tensor4D heatmap = f.gradCAM(modelo, amostra, rotulo);
      f.desenharImagem(heatmap, 15, false, "Heatmap");

      // f.desenharImagem(heatmap.sub(amostra), 15, false, "Heatmap + Amostra");
      // f.desenharImagem(amostra, 15, false, "Amostra");
      
      // f.desenharMnist(modelo);

      // Tensor4D prev = modelo.forward(amostra);
      // prev.reformatar(10, 1);
      // prev.print(4);
      // System.out.println("Dígito " + digito + ", Previsto: " + f.maiorIndice(prev.paraArray()));

      // double ec = f.entropiaCondicional(prev.paraArray());
      // System.out.println("Entropia condicional: " + String.format("%.2f", (100 - (ec * 100))));

      // boolean normalizar = true;
      // f.exportarAtivacoes(modelo, 0, normalizar, 20);
      // f.exportarFiltros(modelo, 0, normalizar, 20);
      // f.exportarAtivacoes(modelo, 3, normalizar, 20);
      // f.exportarFiltros(modelo, 3, normalizar, 20);
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
      Tensor4D amostra = f.carregarImagemCinza("/mnist/teste/" + caminhoImagem + extensao);
      modelo.forward(amostra);
      double[] previsao = modelo.saidaParaArray();
      
      System.out.print("\nTestando: " + caminhoImagem + extensao);
      if(prob){
         System.out.println();
         for(int i = 0; i < previsao.length; i++){
            System.out.println("Prob: " + i + ": " + (int)(previsao[i]*100) + "%");
         }
      }else{
         System.out.print(" -> Prev: " + f.maiorIndice(previsao));
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
            Tensor4D img = f.carregarImagemCinza(caminhoImagem);
            
            modelo.forward(img);
            double[] previsoes = modelo.saidaParaArray();
            if(f.maiorIndice(previsoes) == digito){
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