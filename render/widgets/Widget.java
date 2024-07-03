package render.widgets;

import java.awt.Dimension;

import javax.swing.JPanel;

/**
 * Widfet básico para reaproveitamento.
 */
public abstract class Widget extends JPanel {

	public final int altura;
	public final int largura;
    
    /**
     * Inicializa o widget de acordo com as dimensões desejadas.
     * @param altura altura do painel.
     * @param largura largura do painel.
     */
    protected Widget(int altura, int largura) {
        this.altura = altura;
        this.largura = largura;
		setPreferredSize(new Dimension(this.largura, this.altura));
    }

}
