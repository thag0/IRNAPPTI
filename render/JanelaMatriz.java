package render;

import javax.swing.JFrame;

import jnn.core.tensor.Tensor;
import render.widgets.MatrizConfusao;

public class JanelaMatriz extends JFrame {
	
	public JanelaMatriz(int altura, int largura, Tensor mc) {
		setTitle("Matriz de Confusão");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		MatrizConfusao painel = new MatrizConfusao(altura, largura, mc);
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
