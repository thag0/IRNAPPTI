package render;

import java.awt.GridLayout;

import javax.swing.JFrame;

import jnn.core.tensor.Tensor;
import jnn.modelos.Sequencial;
import render.widgets.DesenhoGrade;
import render.widgets.Previsoes;

public class JanelaDesenho extends JFrame {

	DesenhoGrade pd;
	Previsoes pp;
	Sequencial modelo;
	
	/**
	 * Inicializa uma janela de desenho para testar o modelo usando
	 * uma grade 28x28 (do conjunto mnist)
	 * @param altura altura da janela (pixels).
	 * @param largura largura da janela (pixels).
	 * @param modelo modelo treinado.
	 */
	public JanelaDesenho(int altura, int largura, Sequencial modelo) {
		pd = new DesenhoGrade(altura, largura/2, 28);
		pp = new Previsoes(altura, largura/2);

		this.modelo = modelo;

		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLayout(new GridLayout(1, 2));
		setTitle("Teste modelo");
		setResizable(false);
		
		add(pd);
		add(pp);

		pack();
		
		setLocationRelativeTo(null);
		setVisible(true);
	}

	public void update() {
		double[][] entrada = new double[pd.tamBloco][pd.tamBloco];
		for (int i = 0; i < entrada.length; i++) {
			for (int j = 0; j < entrada[i].length; j++) {
				entrada[i][j] = (pd.blocos[i][j]) ? 1.0 : 0.0;
			}
		}

		Tensor amostra = new Tensor(entrada);
		amostra.unsqueeze(0);//2d -> 3d
		Tensor prev = modelo.forward(amostra);

		double max = prev.maximo().item();
		double[] arr = prev.paraArrayDouble();
		int idMaior = 0, idSegMaior = 0;

		for (int i = 1; i < arr.length; i++) {
			if (arr[i] == max) idMaior = i;
		}

		for (int i = 1; i < arr.length; i++) {
			if (arr[i] < max && arr[i] > arr[idSegMaior]) {
				idSegMaior = i;
			}
		}

		pp.update(
			Integer.toString(idMaior) + " (" + ((int)(arr[idMaior] * 100)) + "%)",
			Integer.toString(idSegMaior) + " (" + ((int)(arr[idSegMaior] * 100)) + "%)"
		);
	}
	
}
