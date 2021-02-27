using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Base
{
    static class Random
    {
        private static System.Random rnd = new System.Random();

        public static float Next(int min, int max)
        {
            return rnd.Next(min, max);
        }

        public static float NextFloat(int min, int max)
        {
            return rnd.Next(min, max) + (float)rnd.NextDouble();
        }
    }
}
