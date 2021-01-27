using System;

namespace NeuralNet
{
    public class Activation
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
            if (x <= 0)
            {
                return 0.01f * (x + 0.000001f);
            }

            return x;
        }

        static public float DerivedReLU(float x)
        {
            if (x < 0)
            {
                return 0.0001f;
            }

            return 0.01f;
        }
    }
}
