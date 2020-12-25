using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NeuralNet;

namespace NeuralNetRun
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();

            float[][] learningData = new float[200][];
            float[][] learningAnswers = new float[200][];

            float[][] testData = new float[20][];
            float[][] testAnswers = new float[20][];

            float lrMin = 100;
            float lrMax = 0;

            for (int i = 0; i < learningData.Length; i++)
            {
                float var = rnd.Next(10, 81);

                learningData[i] = new float[] { var };
                if (learningData[i][0] <= 40) learningAnswers[i] = new float[] { 0 };
                if (learningData[i][0] > 40) learningAnswers[i] = new float[] { 1 };

                if (learningData[i][0] < lrMin) lrMin = learningData[i][0];
                if (learningData[i][0] > lrMax) lrMax = learningData[i][0];
            }

            for (int i = 0; i < testData.Length; i++)
            {
                float var = rnd.Next(10, 81);

                testData[i] = new float[] { var };
                if (testData[i][0] <= 40) testAnswers[i] = new float[] { 0 };
                if (testData[i][0] > 40) testAnswers[i] = new float[] { 1 };
            }

            for (int i = 0; i < learningData.Length; i++)
            {
                learningData[i][0] = Normalize.Minimax(learningData[i][0], lrMin, lrMax);
            }

            for (int i = 0; i < testData.Length; i++)
            {
                testData[i][0] = Normalize.Minimax(testData[i][0], lrMin, lrMax);
            }

            FeedForwardNN nn = new FeedForwardNN(new int[] { 1, 3, 2, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            nn.Train(5, 1000, 0.1f, 0.3f,
                learningData, testData,
                learningAnswers, testAnswers);

            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(39, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(40, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(41, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(42, lrMin, lrMax) })[0]);

            Console.ReadLine();
        }
    }
}
