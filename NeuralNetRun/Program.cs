using NeuralNet.Base;
using NeuralNet.BackPropogation;
using NeuralNet.Genetic;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace NeuralNetRun
{
    public class Program
    {
        static void Main(string[] args)
        {
            PopulationController populationController = new PopulationController
                (new FeedForwardNNDescriptor(new int[] { 1, 8, 8, 16, 8, 4 }, Activations.ReLU, Activations.DerivedReLU),
                50);

            //a+2b+3c+4d=30

            for (int iter = 0; iter < 1000; iter++)
            {
                Console.WriteLine($"=======================GEN {iter + 1}=======================");

                float minRes = 30;

                for (int i = 0; i < populationController.PopulationCount; i++)
                {
                    var u = populationController.GetPopulationUnit(i);

                    float[] Outputs = u.NNRun(new float[] { 1 });

                    float result = Outputs[0] + 2 * Outputs[1] + 3 * Outputs[2] + 4 * Outputs[3];
                    result = Math.Abs(result - 30);

                    u.Rate = result;

                    if (result < minRes)
                    {
                        minRes = result;
                    }

                    Console.WriteLine(result);
                }

                Console.WriteLine($"MIN RESULT: {minRes}");

                populationController.CreateNewPopulation(0.01f, 3, 0.15f, findMin: true);

                var u0 = populationController.GetPopulationUnit(0);
                var u1 = populationController.GetPopulationUnit(1);

                bool weightsIndent = true;

                for (int k = 0; k < u0.NN.Weights.Count; k++)
                {
                    for (int j = 0; j < u0.NN.Weights[k].Count; j++)
                    {
                        for (int l = 0; l < u0.NN.Weights[k][j].Count; l++)
                        {
                            if (u0.NN.Weights[k][j][l] != u1.NN.Weights[k][j][l] && weightsIndent)
                            {
                                weightsIndent = false;
                                break;
                            }
                        }
                    }
                }

                if (weightsIndent)
                {
                    weightsIndent = false;
                }
            }

            Console.ReadLine();
        }
    }
}