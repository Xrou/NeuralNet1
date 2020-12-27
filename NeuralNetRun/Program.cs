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
        }
    }
}
