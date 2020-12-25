﻿using System;

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
    }
}
