namespace NeuralNet.Base
{
    public class Normalize
    {
        public static float Minimax(float var, float min, float max)
        {
            return (var - min) / (max - min);
        }

        public static float ReverseMinimax(float var, float min, float max)
        {
            /*
            (x-min) / (max-min) = 0.53
            (x-13) / (130-13) = 0.53
            (x-13) / 117 = 0.53
            x - 13 = 117 * 0.53
            x - 13 = 62.01
            x = 62.01 + 13
            x = 75.01
            */

            float maxMinusMin = max - min;
            float xMinusMin = var * maxMinusMin;
            float x = xMinusMin + min;

            return x;
        }

        public static float Map(float var, float fromMin, float fromMax, float toMin, float toMax)
        {
            return (var - fromMin) * (toMax - toMin) / (fromMax - fromMin) + toMin;
        }


        public static void ApplyMinimax(ref float[][] arr, float min, float max)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    arr[i][k] = Minimax(arr[i][k], min, max);
                }
            }
        }

        public static void ApplyReverseMinimax(ref float[][] arr, float min, float max)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    arr[i][k] = ReverseMinimax(arr[i][k], min, max);
                }
            }
        }

        public static void ApplyMap(ref float[][] arr, float fromMin, float fromMax, float toMin, float toMax)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                for (int k = 0; k < arr[i].Length; k++)
                {
                    arr[i][k] = Map(arr[i][k], fromMin, fromMax, toMin, toMax);
                }
            }
        }
    }
}
