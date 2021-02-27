using NeuralNet.Base;
using NeuralNet.BackPropogation;
using NeuralNet.Genetic;
using System;
using System.Threading.Tasks;


namespace NeuralNetRun
{
    public class Program
    {
        static void Main(string[] args)
        {
            PopulationController populationController = new PopulationController
                (new FeedForwardNNDescriptor(new int[] { 4, 8, 8, 16, 8, 2 }, Activations.Sigmoid, Activations.DerivedSigmoid),
                50);

            populationController.CreateNewPopulation(0.01f, 1, 0.15f);
        }
    }
}