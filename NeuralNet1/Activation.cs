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

        static public float Tangh(float x)
        {
            return (float)((Math.Pow(E, x) - Math.Pow(E, -x)) / (Math.Pow(E, x) + Math.Pow(E, -x)));
        }

        static public float DerivedTangh(float x)
        {
            return (float)(1 - Math.Pow(Tangh(x), 2));
        }

        static public float ReLU(float x)
        {
            if (x > 0) { return x; }

            else { return x * 0.01f; }
        }

        static public float DerivedReLU(float x)
        {
            if (x >= 0) { return 1; }

            else { return 0.01f; }
        }
    }
}
