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
                (new FeedForwardNNDescriptor(new int[] { 5, 16, 16, 16, 16, 16, 2 }, Activations.Sigmoid, Activations.Sigmoid),
                50);

            float DataMin = Utils.FindMin(Data.testDataNet, Data.testAnswersNet, Data.learnDataNet, Data.learnAnswersNet);
            float DataMax = Utils.FindMax(Data.testDataNet, Data.testAnswersNet, Data.learnDataNet, Data.learnAnswersNet);

            Normalize.ApplyMinimax(ref Data.learnDataNet, DataMin, DataMax);
            Normalize.ApplyMinimax(ref Data.learnAnswersNet, DataMin, DataMax);
            Normalize.ApplyMinimax(ref Data.testDataNet, DataMin, DataMax);
            Normalize.ApplyMinimax(ref Data.testAnswersNet, DataMin, DataMax);
            
            populationController.StartEvolution(10000, Data.learnDataNet, Data.learnAnswersNet,
            (int iter, PopulationUnit best1) =>
            {
                Console.WriteLine($"GEN {iter + 1}");
                Console.WriteLine($"MIN = {best1.Rate}\n");
            }, bestMin: true);


            var best = populationController.GetBest(bestMin: true);
            float[][] o = new float[3][];

            o[0] = best.NNRun(Data.testDataNet[8]);
            o[1] = best.NNRun(Data.testDataNet[3]);
            o[2] = best.NNRun(Data.testDataNet[4]);

            Normalize.ApplyReverseMinimax(ref o, DataMin, DataMax);
            Normalize.ApplyReverseMinimax(ref Data.testDataNet, DataMin, DataMax);
            Normalize.ApplyReverseMinimax(ref Data.testAnswersNet, DataMin, DataMax);

            best.NN.SaveWeights("MY BEEEEEEEEEEEEEEEEST NN.xml");
        }
    }
}