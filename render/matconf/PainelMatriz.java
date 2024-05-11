package render.matconf;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JPanel;

import jnn.core.tensor.Tensor;

public class PainelMatriz extends JPanel {
	
    /**
     * Matriz que será desenhada
     */
    private int[][] matriz;

    /**
     * Espaçamento entre as bordas da janela.
     */
	final int pad = 30;

    // cores
    Color corFundo = new Color(30, 30, 30);
	Color corTexto = new Color(255, 255, 255);
	Color corBorda = new Color(255, 255, 255);
	Color corZero  = new Color(30, 30, 30);
	Color corMin   = new Color(60, 30, 40);
	Color corMax   = new Color(180, 30, 90);

    /**
     * Inicializa um novo painel que desenha uma matriz de confusão.
     * @param altura altura desejada do painel.
     * @param largura largura desejada do painel.
     * @param mc matriz que será desenhada.
     */
	public PainelMatriz(int altura, int largura, Tensor mc) {
		setPreferredSize(new Dimension(largura, altura));
		setBackground(corFundo);

		if (mc.numDim() != 2) {
			throw new IllegalArgumentException(
				"\nTensor deve ser 2D."
			);
		}
		
        int[] shape = mc.shape();
        int alt = shape[0];
        int larg = shape[1];
        matriz = new int[shape[0]][shape[1]];

        for (int i = 0; i < alt; i++) {
            for (int j = 0; j < larg; j++) {
                matriz[i][j] = (int)mc.get(i, j);
            }
        }

	}

	@Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        int alturaUtil = getHeight() - (2 * pad);
        int larguraUtil = getWidth() - (2 * pad);
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

        g.setFont(getFont().deriveFont(16f));

        for (int i = 0; i < linhas; i++) {
            for (int j = 0; j < colunas; j++) {
                int x = pad + (larguraUtil / colunas) * j;
                int y = pad + (alturaUtil / linhas) * i;
                int tamanhoBloco = Math.min(larguraUtil / colunas, alturaUtil / linhas);

                float valNorm = (float) (matriz[i][j] - valorMin) / (valorMax - valorMin);

                Color c = valNorm == 0 ? corZero : interpolarCores(corMin, corMax, valNorm);
                g.setColor(c);
                g.fillRect(x, y, tamanhoBloco, tamanhoBloco);

                g.setColor(corTexto);
                FontMetrics fm = g.getFontMetrics();
                int textoX = x + (tamanhoBloco - fm.stringWidth(String.valueOf(matriz[i][j]))) / 2;
                int textoY = y + (tamanhoBloco - fm.getHeight()) / 2 + fm.getAscent();
                g.drawString(String.valueOf(matriz[i][j]), textoX, textoY);
            }
        }

		// linhas em volta da matriz
		Graphics2D g2 = (Graphics2D) g;
		g2.setStroke(new BasicStroke(3f));
		g.setColor(corBorda);
		g.drawLine(pad, pad, getWidth()-pad, pad);
		g.drawLine(pad, getHeight()-pad, getWidth()-pad, getHeight()-pad);
		g.drawLine(pad, pad, pad, getHeight()-pad);
		g.drawLine(getWidth()-pad, pad, getWidth()-pad, getHeight()-pad);
		g2.setStroke(new BasicStroke(1));

		//texto "Previsto"
		g.setColor(corTexto);
		String txt = "Previsto";
		FontMetrics fm = g.getFontMetrics();
		int txtX = (larguraUtil - fm.stringWidth(txt)) / 2 + pad;
		g.drawString(txt, txtX, getHeight() - 10);
		
		//texto "Real"
		String txtReal = "Real";
		FontMetrics fmReal = g.getFontMetrics();
		int txtRealY = getHeight() / 2;
		int txtRealX = pad - fmReal.stringWidth(txtReal) / 2; 
		g2.rotate(-Math.PI / 2, txtRealX, txtRealY);
		g.drawString(txtReal, txtRealX-16, txtRealY+8);// gambiarra pra alinhar
    }

    /**
     * calcula um novo valor de cor intermediário.
     * @param corMin valor mínimo da cor.
     * @param corMax valor máximo da cor.
     * @param valor escala de interpolação.
     * @return cor interpolada.
     */
    private Color interpolarCores(Color corMin, Color corMax, float valor) {
        int r = (int) (corMin.getRed() + valor * (corMax.getRed() - corMin.getRed()));
        int g = (int) (corMin.getGreen() + valor * (corMax.getGreen() - corMin.getGreen()));
        int b = (int) (corMin.getBlue() + valor * (corMax.getBlue() - corMin.getBlue()));
        return new Color(r, g, b);
    }
}
