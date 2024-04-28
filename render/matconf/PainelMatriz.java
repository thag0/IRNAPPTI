package render.matconf;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;

import javax.swing.JPanel;

public class PainelMatriz extends JPanel{
	private int[][] matriz;

	Color corMax   = new Color(200, 40, 80);
	Color corMin   = new Color(60, 30, 40);
	Color corZero  = new Color(30, 30, 30);
	Color corTexto = new Color(255, 255, 255);

	public PainelMatriz(int altura, int largura, int[][] m) {
		setPreferredSize(new Dimension(largura, altura));
		this.matriz = m;
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		int alturaPainel = getHeight();
		int larguraPainel = getWidth();

		int linhas = matriz.length;
		int colunas = matriz[0].length;

		int valorMax = Integer.MIN_VALUE;
		int valorMin = Integer.MAX_VALUE;
		for (int i = 0; i < linhas; i++) {
			for (int j = 0; j < colunas; j++) {
				valorMax = Math.max(valorMax, matriz[i][j]);
				valorMin = Math.min(valorMin, matriz[i][j]);
			}
		}

		g.setFont(getFont().deriveFont(14.f));

		for (int i = 0; i < linhas; i++) {
			for (int j = 0; j < colunas; j++) {
				int x = (larguraPainel / colunas) * j;
				int y = (alturaPainel / linhas) * i;
				float valorNormalizado = (float) (matriz[i][j] - valorMin) / (valorMax - valorMin);

				Color c = valorNormalizado == 0 ? corZero : interpolarCores(corMin, corMax, valorNormalizado);
				g.setColor(c);
				g.fillRect(x, y, larguraPainel / colunas, alturaPainel / linhas);

				g.setColor(corTexto);
				FontMetrics fm = g.getFontMetrics();
				int textoX = x + (larguraPainel / colunas - fm.stringWidth(String.valueOf(matriz[i][j]))) / 2;
				int textoY = y + ((alturaPainel / linhas) - fm.getHeight()) / 2 + fm.getAscent();
				g.drawString(String.valueOf(matriz[i][j]), textoX, textoY);
			}
		}
	}

	private Color interpolarCores(Color corMin, Color corMax, float valor) {
		int r = (int) (corMin.getRed() + valor * (corMax.getRed() - corMin.getRed()));
		int g = (int) (corMin.getGreen() + valor * (corMax.getGreen() - corMin.getGreen()));
		int b = (int) (corMin.getBlue() + valor * (corMax.getBlue() - corMin.getBlue()));
		return new Color(r, g, b);
	}

}
