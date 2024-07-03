package render.widgets;

import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;

/**
 * Painel de previsões do modelo.
 */
public class Previsoes extends Widget {
	
	/**
	 * Texto para previsão com maior valor.
	 */
	private String txt1 = "";

	/**
	 * Texto para previsão com segundo maior valor.
	 */
	private String txt2 = "";

	/**
	 * Espaçamento entre os textos.
	 */
	final int pad = 30;

	/**
	 * Inicializa o painel de previsões do modelo.
     * @param altura altura desejada do painel.
     * @param largura largura desejada do painel.
	 */
	public Previsoes(int altura, int largura) {
		super(altura, largura);
		setBackground(new Color(24, 24, 24));
	}

	/**
	 * Atualiza os textos do painel.
	 * @param txt1 Texto para previsão com maior valor.
	 * @param txt2 Texto para previsão com segundo maior valor.
	 */
	public void update(String txt1, String txt2) {
		this.txt1 = updateTxt(txt1);
		this.txt2 = updateTxt(txt2);
		repaint();
	}

	/**
	 * Trata o valor de texto.
	 * @param txt texto desejado.
	 */
	private String updateTxt(String txt) {
		if (txt != null) {
			return txt.trim();
		}

		return "";
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
