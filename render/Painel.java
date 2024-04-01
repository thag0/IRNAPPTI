package render;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

import rna.core.Mat;

public class Painel extends JPanel{

   final int escala;
   public final int altura;
   public final int largura;

   Mat mat;

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
    * Desenha o conteúdo da matriz em escala de cinza.
    * @param m matriz.
    */
   public void desenharMat(Mat m){
      if(m == null){
         throw new IllegalArgumentException(
            "\nMatriz não pode ser nula."
         );
      }
      mat = m;
      repaint();
   }

   @Override
   protected void paintComponent(Graphics g){
      super.paintComponent(g);
      Graphics2D g2 = (Graphics2D) g;

      int linhas = mat.lin();
      int colunas = mat.col();

      int larguraPixel = largura / colunas;
      int alturaPixel = altura / linhas;

      for(int i = 0; i < linhas; i++){
         for(int j = 0; j < colunas; j++){
            double valor = mat.elemento(i, j);
            int cinza = (int) (255 * valor);

            if(cinza > 255) cinza = 255;
            if(cinza < 0) cinza = 0;

            int verm = cinza;
            int verd = cinza;
            int azul = cinza;
            g2.setColor(new Color(verm, verd, azul));

            int x = j * larguraPixel;
            int y = i * alturaPixel;

            g2.fillRect(x, y, larguraPixel, alturaPixel);
         }
      }
   }

}
