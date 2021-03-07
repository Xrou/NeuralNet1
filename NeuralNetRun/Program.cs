using NeuralNet.Base;
using NeuralNet.BackPropogation;
using NeuralNet.Genetic;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using NeuralNet.DataSets;
using System.IO;

namespace NeuralNetRun
{
    public class Program
    {
        static void Main(string[] args)
        {
            FeedForwardNN nn = new FeedForwardNN(new FeedForwardNNDescriptor());
            nn.ReadWeights("mnist.xml");

            float[][] testData = new float[10000][];
            float[][] testAnswers = new float[10000][];

            int index = 0;

            using (StreamReader sr = new StreamReader(@"C:\Users\Dima\source\repos\NeuralNet1\NeuralNetRun\bin\Debug\mnist_test.csv", System.Text.Encoding.Default))
            {
                string line;

                while ((line = sr.ReadLine()) != null)
                {
                    string[] array = line.Split(',');
                    float[] data = new float[784];
                    int label = Convert.ToInt32(array[0]);

                    for (int i = 0; i < 784; i++)
                    {
                        data[i] = Normalize.Minimax(Convert.ToInt32(array[i + 1]), 0, 255);
                    }

                    float[] answer = new float[10];
                    answer[label] = 1;

                    testData[index] = data;
                    testAnswers[index] = answer;
                    index++;
                }
            }

            Trainer trainer = new Trainer(ref nn);

            trainer.TrainBackPropogation(1, 1, 0.0001f, 0.3f, testData, new float[][] { }, testAnswers, new float[][] { });

            float[][] o = new float[3][];

            o[0] = nn.Run(testData[0]);
            o[1] = nn.Run(testData[1]);
            o[2] = nn.Run(testData[2]);

            int answer0;
            int answer1;
            int answer2;

            float max = 0;

            for (int i = 0; i < o[0].Length; i++)
            {
                if (o[0][i] > max)
                {
                    answer0 = i;
                    max = o[0][i];
                }
            }

            max = 0;

            for (int i = 0; i < o[0].Length; i++)
            {
                if (o[1][i] > max)
                {
                    answer1 = i;
                    max = o[1][i];
                }
            }

            max = 0;

            for (int i = 0; i < o[0].Length; i++)
            {
                if (o[2][i] > max)
                {
                    answer2 = i;
                    max = o[2][i];
                }
            }

        }
    }
}