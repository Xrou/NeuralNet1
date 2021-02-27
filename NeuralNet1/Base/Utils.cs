namespace NeuralNet.Base
{
    public class Utils
    {
        public static float FindMin(params float[][][] arr)
        {
            float min = arr[0][0][0];

            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    for (int j = 0; j < arr[i][k].Length; j++)
                    {
                        if (arr[i][k][j] < min) min = arr[i][k][j];
                    }
                }
            }

            return min;
        }

        public static float FindMax(params float[][][] arr)
        {
            float max = arr[0][0][0];

            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    for (int j = 0; j < arr[i][k].Length; j++)
                    {
                        if (arr[i][k][j] > max) max = arr[i][k][j];
                    }
                }
            }

            return max;
        }
    }
}
