package render;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

import jnn.core.Tensor4D;

public class Painel extends JPanel{

   final int escala;
   public final int altura;
   public final int largura;

   Tensor4D tensor;

   /**
    * Cria um novo painel com base nos valores dados.
    * @param altura altura do painel.
    * @param largura largura do painel.
    * @param escala escala de ampliação (altura e largura).
    */
   public Painel(int altura, int largura, int escala){
      this.escala = escala;
      this.altura =  escala * altura;
      this.largura = escala * largura;

      setPreferredSize(new Dimension(this.largura, this.altura));
      setBackground(Color.black);
   }
   
   /**
    * Desenha o conteúdo 2D do tensor.
    * @param tensor tensor desejado.
    */
   public void desenharImagem(Tensor4D tensor){
      if(tensor == null){
         throw new IllegalArgumentException(
            "\nMatriz não pode ser nula."
         );
      }
      this.tensor = tensor;
      repaint();
   }

   @Override
   protected void paintComponent(Graphics g){
      super.paintComponent(g);
      Graphics2D g2 = (Graphics2D) g;

      int alt = tensor.dim3();
      int larg = tensor.dim4();

      int largPixel = largura / larg;
      int altPixel  = altura / alt;

      boolean rgb = tensor.dim2() == 3;

      for(int i = 0; i < alt; i++){
         for(int j = 0; j < larg; j++){

            if(rgb){
               double valR = tensor.get(0, 0, i, j);
               double valG = tensor.get(0, 1, i, j);
               double valB = tensor.get(0, 2, i, j);
               
               int corR = (int)(valR * 255);
               if(corR > 255) corR = 255;
               if(corR < 0) corR = 0;
               
               int corG = (int)(valG * 255);
               if(corG > 255) corG = 255;
               if(corG < 0) corG = 0;
               
               int corB = (int)(valB * 255);
               if(corB > 255) corB = 255;
               if(corB < 0) corR = 0;
               
               g2.setColor(new Color(corR, corG, corB));
            }else{
               double valCinza = tensor.get(0, 0, i, j);
               int cinza = (int)(valCinza * 255);
               if(cinza > 255) cinza = 255;
               if(cinza < 0) cinza = 0;
               g2.setColor(new Color(cinza, cinza, cinza));
            }

            int x = j * largPixel;
            int y = i * altPixel;

            g2.fillRect(x, y, largPixel, altPixel);
         }
      }
   }

}
