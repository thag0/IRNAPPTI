import ged.Ged;
import rna.modelos.Sequencial;
import rna.serializacao.Serializador;

public class Teste {
   static final String CAMINHO_MODELO = "./modelos/rna/";
   static Ged ged = new Ged();

   public static void main(String[] args) {
      ged.limparConsole();

      String nomeModelo = "conv-mnist-93-6";
      Sequencial modelo = new Serializador().lerSequencial(CAMINHO_MODELO + nomeModelo + ".txt");
      modelo.info();
   }
}
