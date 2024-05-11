package render.matconf;

import javax.swing.JFrame;

import jnn.core.tensor.Tensor;

public class JanelaMatriz extends JFrame {
	
	public JanelaMatriz(int altura, int largura, Tensor mc) {
		setTitle("Matriz de Confusão");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		PainelMatriz painel = new PainelMatriz(altura, largura, mc);
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
