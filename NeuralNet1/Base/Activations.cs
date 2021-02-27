using System;

namespace NeuralNet.Base
{
    public class Activations
    {
        private const float E = 2.7182818285f;

        static public float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Pow(E, -x)));
        }

        static public float DerivedSigmoid(float x)
        {
            return (1 - x) * x;
        }

        static public float ReLU(float x)
        {
            if (x < 0)
            {
                return 0.1f * x;
            }

            return x;
        }

        static public float DerivedReLU(float x)
        {
            if (x < 0)
            {
                return 0.1f;
            }

            return 1;
        }

        static public float Tanh(float x)
        {
            return (float)((Math.Pow(E, x) - Math.Pow(E, -x)) / (Math.Pow(E, x) + Math.Pow(E, -x)));
        }

        static public float DeriverTanh(float x)
        {
            return (float)(1 - Math.Pow(Tanh(x), 2));
        }
    }
}
