package render.realtime;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import javax.swing.JPanel;

public class PainelDesenho extends JPanel {

	public final int tamBloco = 28;
	public boolean[][] blocosPintados = new boolean[tamBloco][tamBloco];

	public PainelDesenho(int altura, int largura) {
		setPreferredSize(new Dimension(largura, altura));
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
					for(int i = 0; i < blocosPintados.length; i++){
						for(int j = 0; j < blocosPintados[i].length; j++){
							blocosPintados[i][j] = false;
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
		int colunaClicada = x / (getWidth() / tamBloco);
		int linhaClicada = y / (getHeight() / tamBloco);
		if (colunaClicada >= 0 && colunaClicada < blocosPintados[0].length &&
					linhaClicada >= 0 && linhaClicada < blocosPintados.length) {
			blocosPintados[linhaClicada][colunaClicada] = true;
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
				if (blocosPintados[i][j]) {
					g.setColor(Color.white);
					g.fillRect(j * tamanhoBloco, i * tamanhoBloco, tamanhoBloco, tamanhoBloco);
				}
			}
		}
	}

}

