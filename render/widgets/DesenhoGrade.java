package render.widgets;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * Painel de desenho com grades usando o mouse.
 */
public class DesenhoGrade extends Widget {

	/**
	 * Quantidade de blocos horizontais e verticais.
	 */
	public final int tamBloco;

	/**
	 * Grade de desenho do painel.
	 */
	public boolean[][] blocos;

	/**
	 * Inicializa o painel de desenho em grade.
	 * @param altura altura desejada do painel.
	 * @param largura largura desejada do painel.
	 * @param tam quantidade de grades (horizontais e verticais).
	 */
	public DesenhoGrade(int altura, int largura, int tam) {
		super(largura, altura);

		if (tam < 1) {
			throw new IllegalArgumentException(
				"\nTamanho da grade deve ser maior que zero."
			);
		}
		tamBloco = tam;
		blocos = new boolean[tamBloco][tamBloco];

		setBackground(Color.BLACK);

		addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				desenharBloco(e.getX(), e.getY());
			}
		});

		addMouseMotionListener(new MouseAdapter() {
			@Override
			public void mouseDragged(MouseEvent e) {
				desenharBloco(e.getX(), e.getY());
			}
		});

		addKeyListener(new KeyListener() {

			@Override
			public void keyTyped(KeyEvent e) {}

			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_R){
					for(int i = 0; i < blocos.length; i++){
						for(int j = 0; j < blocos[i].length; j++){
							blocos[i][j] = false;
						}
					}
					repaint();
				}
			}

			@Override
			public void keyReleased(KeyEvent e) {}
			
		});

		setFocusable(true);
	}

	private void desenharBloco(int x, int y) {
		int col = x / (getWidth() / tamBloco);
		int lin = y / (getHeight() / tamBloco);
		if ((col >= 0 && col < blocos[0].length) && (lin >= 0 && lin < blocos.length)) {
			blocos[lin][col] = true;
			repaint();
		}
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
	
		int tamanhoBloco = Math.min(getWidth() / tamBloco, getHeight() / tamBloco);
	
		for (int i = 0; i <= tamBloco; i++) {
			int x = i * tamanhoBloco;
			g.drawLine(x, 0, x, tamBloco * tamanhoBloco);
			int y = i * tamanhoBloco;
			g.drawLine(0, y, tamBloco * tamanhoBloco, y);
		}
	
		for (int i = 0; i < tamBloco; i++) {
			for (int j = 0; j < tamBloco; j++) {
				if (blocos[i][j]) {
					g.setColor(Color.white);
					g.fillRect(j * tamanhoBloco, i * tamanhoBloco, tamanhoBloco, tamanhoBloco);
				}
			}
		}
	}

}

