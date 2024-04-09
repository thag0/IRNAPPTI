package render.realtime;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import javax.swing.JPanel;

public class PainelPrevisoes extends JPanel{
   
   String texto = "";

   public PainelPrevisoes(int altura, int largura) {
      setPreferredSize(new Dimension(largura, altura));
      setBackground(new Color(24, 24, 24));
   }

   @Override
   protected void paintComponent(Graphics g) {
      super.paintComponent(g);

      texto = "Previsto: " + texto;
      g.setColor(Color.white);
      g.setFont(getFont().deriveFont(40.f));

      // Obtém as métricas do texto
      FontMetrics metrics = g.getFontMetrics();
      int larguraTexto = metrics.stringWidth(texto);
      int alturaTexto = metrics.getHeight();

      // Calcula a posição de desenho para centralizar o texto
      int x = (getWidth() - larguraTexto) / 2;
      int y = (getHeight() - alturaTexto) / 2 + metrics.getAscent();

      // Desenha o texto centralizado
      g.drawString(texto, x, y);
   }
}
