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

   public Painel(int altura, int largura, int escala){
      this.escala = escala;
      this.altura =  escala * altura;
      this.largura = escala * largura;

      setPreferredSize(new Dimension(this.largura, this.altura));
      setBackground(Color.black);
   }
   
   public void desenharMat(Mat m){
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

            g2.setColor(new Color(cinza, cinza, cinza));

            int x = j * larguraPixel;
            int y = i * alturaPixel;

            g2.fillRect(x, y, larguraPixel, alturaPixel);
         }
      }
   }

}
