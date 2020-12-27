using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Normalize
    {
        public static float Minimax(float var, float min, float max)
        {
            return (var - min) / (max - min);
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
    }
}
