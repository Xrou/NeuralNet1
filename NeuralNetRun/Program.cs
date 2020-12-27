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
            float min = Utils.FindMin(Data.learnDataFnet, Data.learnDataJnet, Data.testDataFnet, Data.testDataJnet, Data.learnAnswersFnet, Data.learnAnswersJnet, Data.testAnswersFnet, Data.testAnswersJnet);
            float max = Utils.FindMax(Data.learnDataFnet, Data.learnDataJnet, Data.testDataFnet, Data.testDataJnet, Data.learnAnswersFnet, Data.learnAnswersJnet, Data.testAnswersFnet, Data.testAnswersJnet);

            Normalize.ApplyMinimax(ref Data.learnDataFnet, min, max);
            Normalize.ApplyMinimax(ref Data.learnDataJnet, min, max);
            Normalize.ApplyMinimax(ref Data.testDataFnet, min, max);
            Normalize.ApplyMinimax(ref Data.testDataJnet, min, max);
            Normalize.ApplyMinimax(ref Data.learnAnswersFnet, min, max);
            Normalize.ApplyMinimax(ref Data.learnAnswersJnet, min, max);
            Normalize.ApplyMinimax(ref Data.testAnswersFnet, min, max);
            Normalize.ApplyMinimax(ref Data.testAnswersJnet, min, max);

            FeedForwardNN FNet = new FeedForwardNN(new int[] { 3, 9, 9, 9, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);
            FeedForwardNN JNet = new FeedForwardNN(new int[] { 4, 9, 9, 9, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            Task FNetThread = new Task(() => FNet.Train(5, 1000, 0.1f, 0.3f, Data.learnDataFnet, Data.testDataFnet, Data.learnAnswersFnet, Data.testAnswersFnet, logFileName: "FNet train log.txt"));
            Task JNetThread = new Task(() => JNet.Train(5, 1000, 0.1f, 0.3f, Data.learnDataJnet, Data.testDataJnet, Data.learnAnswersJnet, Data.testAnswersJnet, logFileName: "JNet train log.txt"));

            FNetThread.Start();
            JNetThread.Start();

            FNetThread.Wait();
            JNetThread.Wait();

            float na11 = Normalize.Minimax(33, min, max);//0.12
            float na12 = Normalize.Minimax(48, min, max);//0.35
            float na21 = Normalize.Minimax(35, min, max);//0.15
            float na22 = Normalize.Minimax(43, min, max);//0.27

            float[] a11 = FNet.Run(new float[] { 35, 43, 26 }); //33
            float[] a12 = FNet.Run(new float[] { 46, 54, 44 }); //48
            float[] a21 = JNet.Run(new float[] { 35, 35, 38, 26 }); //35
            float[] a22 = JNet.Run(new float[] { 46, 49, 50, 44 }); //43
        }
    }
}
