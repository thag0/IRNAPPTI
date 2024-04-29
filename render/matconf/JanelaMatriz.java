package render.matconf;

import javax.swing.JFrame;

public class JanelaMatriz extends JFrame {
	
	public JanelaMatriz(int altura, int largura, int[][] m) {
		setTitle("Matriz de Confusão");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		PainelMatriz painel = new PainelMatriz(altura, largura, m);
		setResizable(false);
		add(painel);
		pack();

		setLocationRelativeTo(null);
		setVisible(false);
	}

	public void exibir() {
		setVisible(true);
	}
}
