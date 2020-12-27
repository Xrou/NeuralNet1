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

        public static float ReverseMinimax(float var, float min, float max)
        {
            return (var - min) / (max - min);
        }
    }
}
