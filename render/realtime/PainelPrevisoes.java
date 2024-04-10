package render.realtime;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import javax.swing.JPanel;

public class PainelPrevisoes extends JPanel{
   
   String txt1 = "";
   String txt2 = "";
   final int pad = 30;

   public PainelPrevisoes(int altura, int largura) {
      setPreferredSize(new Dimension(largura, altura));
      setBackground(new Color(24, 24, 24));
   }

   @Override
   protected void paintComponent(Graphics g) {
      super.paintComponent(g);

      g.setColor(Color.white);
      g.setFont(getFont().deriveFont(30.f).deriveFont(1));

      FontMetrics metrics = g.getFontMetrics();
      int larguraTexto = metrics.stringWidth(txt1);
      int alturaTexto = metrics.getHeight();

      int x = (getWidth() - larguraTexto) / 2;
      int y = (getHeight() - alturaTexto) / 2 + metrics.getAscent();
      
      g.drawString(txt1, x, y-pad);
      g.drawString(txt2, x, y+pad);
   }
}
