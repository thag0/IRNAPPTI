package render.realtime;

import java.awt.GridLayout;

import javax.swing.JFrame;

import rna.core.Tensor4D;
import rna.modelos.Modelo;

public class JanelaDesenho extends JFrame{

   PainelDesenho pd;
   PainelPrevisoes pp;
   Modelo modelo;
   
   public JanelaDesenho(int altura, int largura, Modelo modelo) {
      pd = new PainelDesenho(altura, largura/2);
      pp = new PainelPrevisoes(altura, largura/2);

      this.modelo = modelo;

      setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      setLayout(new GridLayout(1, 2));
      setTitle("Teste modelo");
      
      add(pd);
      add(pp);
      
      setVisible(true);
      pack();

      setLocationRelativeTo(null);
   }

   public void atualizar(){
      double[][] entrada = new double[pd.tamBloco][pd.tamBloco];
      for(int i = 0; i < entrada.length; i++){
         for(int j = 0; j < entrada[i].length; j++){
            entrada[i][j] = (pd.blocosPintados[i][j] == true) ? 1.0 : 0.0;
         }
      }

      Tensor4D amostra = new Tensor4D(entrada);
      Tensor4D prev = modelo.forward(amostra);

      double max = prev.maximo();
      double[] arr = prev.paraArray();
      int id = 0;
      for(int i = 1; i < arr.length; i++){
         if(arr[i] == max) id = i;
      }

      pp.texto = Integer.toString(id) + "(" + ((int)(arr[id] * 100)) + "%)";
      pp.repaint();
   }
}
