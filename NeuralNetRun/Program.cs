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
            float[][] Train = new float[2000][];
            float[][] TrainAnswers = new float[2000][];
            float[][] Test = new float[2000][];
            float[][] TestAnswers = new float[2000][];

            Random rnd = new Random();

            for (int i = 0; i < 2000; i++)
            {
                float emp = rnd.Next(-7, 8);

                Train[i] = new float[3];
                Test[i] = new float[3];

                Train[i][0] = 20 + emp;
                Train[i][1] = 68 + emp * 1.4f;
                Train[i][2] = (Train[i][0] + Train[i][1]) * 1.4f - 3;

                Test[i][0] = 20 + emp;
                Test[i][1] = 68 + emp * 1.4f;
                Test[i][2] = (Test[i][0] + Test[i][1]) * 1.4f - 3;
            }

            for (int i = 0; i < 2000; i++)
            {
                TrainAnswers[i] = new float[1];
                TestAnswers[i] = new float[1];

                TrainAnswers[i][0] = (Train[i][0] * 1.3f + Train[i][1] * 1.1f + Train[i][2] * 0.7f) * 0.45f;
                TestAnswers[i][0] = (Test[i][0] * 1.3f + Test[i][1] * 1.1f + Test[i][2] * 0.7f) * 0.45f;
            }


            float min = Utils.FindMin(Train, TrainAnswers, Test, TestAnswers); //ищем минимум и максимум из всех данных
            float max = Utils.FindMax(Train, TrainAnswers, Test, TestAnswers);

            Normalize.ApplyMinimax(ref Train, min, max);
            Normalize.ApplyMinimax(ref TrainAnswers, min, max);
            Normalize.ApplyMinimax(ref Test, min, max);
            Normalize.ApplyMinimax(ref TestAnswers, min, max);

            FeedForwardNN FNet = new FeedForwardNN(new int[] { 3, 18, 16, 10, 5, 4, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            FNet.Train(3, 500, 0.0001f, 0.2f, Train, Test, TrainAnswers, TestAnswers, new float[] { 0.3f, 0.2f, 0.15f, 0.1f, 0.1f });

            float val1 = FNet.Run(Test[0])[0];
            float val2 = FNet.Run(Test[1])[0];
            float val3 = FNet.Run(Test[2])[0];

            float a1 = Normalize.ReverseMinimax(TestAnswers[0][0], min, max);
            float a2 = Normalize.ReverseMinimax(TestAnswers[1][0], min, max);
            float a3 = Normalize.ReverseMinimax(TestAnswers[2][0], min, max);

            float revNorm1 = Normalize.ReverseMinimax(val1, min, max);
            float revNorm2 = Normalize.ReverseMinimax(val2, min, max);
            float revNorm3 = Normalize.ReverseMinimax(val3, min, max);
        }
    }
}


/* main
 
            float min = Utils.FindMin(Data.learnDataFnet, Data.learnDataJnet, Data.testDataFnet, Data.testDataJnet, Data.learnAnswersFnet, Data.learnAnswersJnet, Data.testAnswersFnet, Data.testAnswersJnet); //ищем минимум и максимум из всех данных
            float max = Utils.FindMax(Data.learnDataFnet, Data.learnDataJnet, Data.testDataFnet, Data.testDataJnet, Data.learnAnswersFnet, Data.learnAnswersJnet, Data.testAnswersFnet, Data.testAnswersJnet);

            Normalize.ApplyMinimax(ref Data.learnDataFnet, min, max); //приводим к диапозону [0, 1]
            Normalize.ApplyMinimax(ref Data.learnDataJnet, min, max);
            Normalize.ApplyMinimax(ref Data.testDataFnet, min, max);
            Normalize.ApplyMinimax(ref Data.testDataJnet, min, max);
            Normalize.ApplyMinimax(ref Data.learnAnswersFnet, min, max);
            Normalize.ApplyMinimax(ref Data.learnAnswersJnet, min, max);
            Normalize.ApplyMinimax(ref Data.testAnswersFnet, min, max);
            Normalize.ApplyMinimax(ref Data.testAnswersJnet, min, max);

            FeedForwardNN FNet = new FeedForwardNN(new int[] { 3, 9, 9, 9, 9, 9, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid); // инициализация сети
            FeedForwardNN JNet = new FeedForwardNN(new int[] { 4, 9, 9, 9, 9, 9, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            Task FNetTask = new Task(() => FNet.Train(5, 1000, 0.01f, 0.2f, Data.learnDataFnet, Data.testDataFnet, Data.learnAnswersFnet, Data.testAnswersFnet, logFileName: "FNet train log.txt")); // создаем таск по обучению
            Task JNetTask = new Task(() => JNet.Train(5, 1000, 0.01f, 0.2f, Data.learnDataJnet, Data.testDataJnet, Data.learnAnswersJnet, Data.testAnswersJnet, logFileName: "JNet train log.txt"));

            FNetTask.Start(); // запускаем обучение в два потока
            JNetTask.Start();

            FNetTask.Wait(); // ждем пока сети обучатся
            JNetTask.Wait();

            // смотрим пример из тестов

            float nF1 = Normalize.Minimax(33, min, max);
            float nF2 = Normalize.Minimax(48, min, max);
            float nJ1 = Normalize.Minimax(35, min, max);
            float nJ2 = Normalize.Minimax(43, min, max);

            float aF1 = FNet.Run(new float[] { 35, 43, 26 })[0];
            float aF2 = FNet.Run(new float[] { 46, 54, 44 })[0];
            float aJ1 = JNet.Run(new float[] { 35, 35, 38, 26 })[0];
            float aJ2 = JNet.Run(new float[] { 46, 49, 50, 44 })[0];
 */


/*
            float[][] A = new float[2000][];
            float[][] B = new float[100][];
            float[][] C = new float[2000][];
            float[][] D = new float[100][];

            Random rnd = new Random();

            for (int i = 0; i < 2000; i++)
            {
                float emp = rnd.Next(-5, 5);

                A[i] = new float[3];
                C[i] = new float[1];

                A[i][0] = 50 + emp;
                A[i][1] = 10 + emp + rnd.Next(-1, 1);
                A[i][2] = 11 + emp + rnd.Next(-1, 1);
                C[i][0] = (50 + 10 + emp) / 2;
            }

            for (int i = 0; i < 100; i++)
            {
                float emp = rnd.Next(-5, 5);

                B[i] = new float[3];
                D[i] = new float[1];

                B[i][0] = 50 + emp;
                B[i][1] = 10 + emp + rnd.Next(-1, 1);
                B[i][2] = 11 + emp + rnd.Next(-1, 1);
                D[i][0] = (50 + 10 + emp) / 2;
            }


            float min = Utils.FindMin(A, C); //ищем минимум и максимум из всех данных
            float max = Utils.FindMax(A, C);

            Normalize.ApplyMinimax(ref A, min, max);
            Normalize.ApplyMinimax(ref C, min, max);

            FeedForwardNN FNet = new FeedForwardNN(new int[] { 3, 9, 9, 9, 9, 9, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            FNet.Train(5, 1000, 0.01f, 0.2f, A, B, C, D);

            float val1 = FNet.Run(B[0])[0];
            float val2 = FNet.Run(B[1])[0];
            float val3 = FNet.Run(B[2])[0];
 */