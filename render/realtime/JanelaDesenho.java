package render.realtime;

import java.awt.GridLayout;

import javax.swing.JFrame;

import jnn.core.tensor.Tensor;
import jnn.modelos.Sequencial;

public class JanelaDesenho extends JFrame {

	PainelDesenho pd;
	PainelPrevisoes pp;
	Sequencial modelo;
	
	/**
	 * Inicializa uma janela de desenho para testar o modelo usando
	 * uma grade 28x28 (do conjunto mnist)
	 * @param altura altura da janela (pixels).
	 * @param largura largura da janela (pixels).
	 * @param modelo modelo treinado.
	 */
	public JanelaDesenho(int altura, int largura, Sequencial modelo) {
		pd = new PainelDesenho(altura, largura/2);
		pp = new PainelPrevisoes(altura, largura/2);

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

	public void atualizar() {
		double[][] entrada = new double[pd.tamBloco][pd.tamBloco];
		for (int i = 0; i < entrada.length; i++) {
			for (int j = 0; j < entrada[i].length; j++) {
				entrada[i][j] = (pd.blocos[i][j]) ? 1.0 : 0.0;
			}
		}

		Tensor amostra = new Tensor(entrada);
		amostra.unsqueeze(0);
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

		pp.txt1 = Integer.toString(idMaior) + " (" + ((int)(arr[idMaior] * 100)) + "%)";
		pp.txt2 = Integer.toString(idSegMaior) + " (" + ((int)(arr[idSegMaior] * 100)) + "%)";
		pp.repaint();
	}
	
}
