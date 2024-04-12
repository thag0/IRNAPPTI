import rna.core.Tensor4D;

public class Teste {
   public static void main(String[] args) {
      double[][] m = {
         {1, 2, 1},
         {2, 3, 2},
         {1, 2, 1},
      };

      m = scaleMat(m, 7, 7);

      Tensor4D tensor = new Tensor4D(m);
      tensor.normalizar(0, 1);
      new Funcional().desenharImagem(tensor, 40, false, "");
   }

   public static double[][] scaleMat(double[][] m, int newWidth, int newHeight) {
      int height = m.length;
      int width = m[0].length;
      
      double[][] scaledMatrix = new double[newHeight][newWidth];
      
      for (int i = 0; i < newHeight; i++) {
         for (int j = 0; j < newWidth; j++) {
            double scaledHeight = (double) i / (newHeight - 1) * (height - 1);
            double scaledWidth = (double) j / (newWidth - 1) * (width - 1);
            
            int y0 = (int) scaledHeight;
            int x0 = (int) scaledWidth;
            int y1 = Math.min(y0 + 1, height - 1);
            int x1 = Math.min(x0 + 1, width - 1);
            
            double dx = scaledWidth - x0;
            double dy = scaledHeight - y0;
            
            double interpolatedValue = 
               (1 - dx) * (1 - dy) * m[y0][x0] +
               dx * (1 - dy) * m[y0][x1] +
               (1 - dx) * dy * m[y1][x0] +
               dx * dy * m[y1][x1];
            
            scaledMatrix[i][j] = interpolatedValue;
         }
      }
      
      return scaledMatrix;
   }
}
