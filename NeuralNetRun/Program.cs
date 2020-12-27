using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using NeuralNet;


namespace NeuralNetRun
{
    public class Program
    {
        static void Main(string[] args)
        {
            float min = 100;
            float max = 0;

            for (int i = 0; i < Data.learnData.Length; i++)
            {
                for (int k = 0; k < Data.learnData[i].Length; k++)
                {
                    if (Data.learnData[i][k] < min) min = Data.learnData[i][k];
                    if (Data.learnData[i][k] > max) max = Data.learnData[i][k];
                }
            }

            for (int i = 0; i < Data.testData.Length; i++)
            {
                for (int k = 0; k < Data.testData[i].Length; k++)
                {
                    if (Data.testData[i][k] < min) min = Data.testData[i][k];
                    if (Data.testData[i][k] > max) max = Data.testData[i][k];
                }
            }

            for (int i = 0; i < Data.learnData.Length; i++)
            {
                for (int k = 0; k < Data.learnData[i].Length; k++)
                {
                    Data.learnData[i][k] = Normalize.Minimax(Data.learnData[i][k], min, max);
                }

                Data.learnAnswers[i][0] = Normalize.Minimax(Data.learnAnswers[i][0], min, max);
                Data.learnAnswers[i][1] = Normalize.Minimax(Data.learnAnswers[i][1], min, max);
            }

            for (int i = 0; i < Data.testData.Length; i++)
            {
                for (int k = 0; k < Data.testData[i].Length; k++)
                {
                    Data.testData[i][k] = Normalize.Minimax(Data.testData[i][k], min, max); 
                }

                Data.testAnswers[i][0] = Normalize.Minimax(Data.testAnswers[i][0], min, max);
                Data.testAnswers[i][1] = Normalize.Minimax(Data.testAnswers[i][1], min, max);
            }

            FeedForwardNN nn = new FeedForwardNN(new int[] { 5, 14, 28, 28, 9, 2 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            nn.Train(3, 1000, 0.01f, 0.35f,
                Data.learnData,
                Data.testData,
                Data.learnAnswers,
                Data.testAnswers,
                new float[] { 0.20f, 0.25f, 0.25f, 0.20f });

            float[][] answ = new float[3][];

            float a1_1 = Normalize.Minimax(66, min, max);//0.63
            float a1_2 = Normalize.Minimax(54, min, max);//0.44
            float a2_1 = Normalize.Minimax(44, min, max);//0.29
            float a2_2 = Normalize.Minimax(54, min, max);//0.44
            float a3_1 = Normalize.Minimax(40, min, max);//0.23
            float a3_2 = Normalize.Minimax(54, min, max);//0.44

            answ[0] = nn.Run(new float[] { 58, 65, 67, 50, 60 }); //66 54
            answ[1] = nn.Run(new float[] { 42, 43, 49, 55, 52 }); //44 54
            answ[2] = nn.Run(new float[] { 49, 62, 58, 50, 60 }); //40 54

            Console.ReadLine();
        }
    }
}
