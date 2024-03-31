package render;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;

import rna.core.Mat;

public class Janela extends JFrame{
   
   Painel painel;
   LeitorTeclado leitorTeclado;

   public Janela(int altura, int largura, int escala, String titulo){
      if(titulo == null) titulo = "Janela";
      else setTitle(titulo);

      this.painel = new Painel(altura, largura, escala);
      add(painel);
      pack();

      setResizable(false);
      setVisible(true);
      setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      setLocationRelativeTo(null);
      
      addWindowListener(new WindowAdapter() {
         @Override
         public void windowClosing(WindowEvent e) {
            dispose();
         }
      });
      
      leitorTeclado = new LeitorTeclado();
      addKeyListener(leitorTeclado); 
      setFocusable(true);
   }

   public Janela(int altura, int largura, String titulo){
      this(altura, largura, 1, titulo);
   }

   public Janela(int altura, int largura, int escala){
      this(altura, largura, escala, "Janela");
   }

   public Janela(int altura, int largura){
      this(altura, largura, 1, "Janela");
   }

   public void desenharMat(Mat mat){
      desenharMat(mat, getTitle());
   }

   public void desenharMat(Mat mat, String titulo){
      if(mat == null){
         throw new IllegalArgumentException(
            "\nMatriz não pode ser nula."
         );
      }
      
      setTitle(titulo);
      painel.desenharMat(mat);
   }

   public void desenharArray(Mat[] arr){
      int indice = 0;
      int tamanho = arr.length;

      while(this.isActive()){
         if(leitorTeclado.d){
            if(indice+1 < tamanho) indice++;
            else indice = 0;
            leitorTeclado.d = false;
         
         }else if(leitorTeclado.a){
            if(indice-1 >= 0) indice--;
            else indice = tamanho-1;
            leitorTeclado.a = false;
         }

         setTitle("Amostra " + indice);
         painel.desenharMat(arr[indice]);

         try{
            Thread.sleep(50);
         }catch(Exception e){
            e.printStackTrace();
         }
      }
   }
}

class LeitorTeclado implements KeyListener{

   boolean w = false, a = false, s = false, d = false;

   public void keyTyped(KeyEvent e){}

   @Override
   public void keyPressed(KeyEvent e){
      switch(e.getKeyCode()){
         case KeyEvent.VK_W: w = true; break;
         case KeyEvent.VK_A: a = true; break;
         case KeyEvent.VK_S: s = true; break;
         case KeyEvent.VK_D: d = true; break;
      }
   }

   @Override
   public void keyReleased(KeyEvent e){
      switch(e.getKeyCode()){
         case KeyEvent.VK_W: w = false; break;
         case KeyEvent.VK_A: a = false; break;
         case KeyEvent.VK_S: s = false; break;
         case KeyEvent.VK_D: d = false; break;
      }
   }
}
