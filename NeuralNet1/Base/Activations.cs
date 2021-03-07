using System;

namespace NeuralNet.Base
{
    public static class Activations
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

        static public float DerivedTanh(float x)
        {
            return (float)(1 - Math.Pow(Tanh(x), 2));
        }

        static public string AsString(Activation a)
        {
            if (a == Sigmoid)
                return "Sigmoid";

            else if (a == ReLU)
                return "ReLU";

            else if (a == Tanh)
                return "Tanh";

            return "";
        }

        static public (Activation, DerivedActivation) FromString(string a)
        {
            if (a == "Sigmoid")
                return (Sigmoid, DerivedSigmoid);

            else if (a == "ReLU")
                return (ReLU, DerivedReLU);

            else if (a == "Tanh")
                return (Tanh, DerivedTanh);

            return (null, null);
        }
    }
}
