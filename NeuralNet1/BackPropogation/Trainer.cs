using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet.Base;
using NeuralNet.DataSets;

namespace NeuralNet.BackPropogation
{
    public class Trainer
    {
        FeedForwardNN NeuralNet;

        public Trainer(ref FeedForwardNN NN)
        {
            NeuralNet = NN;
        }

        public int TrainBackPropogation(int epochs, int iterCount, float learningRate, float moment, float[][] learnData, float[][] testData, float[][] learnAnswers, float[][] testAnswers, float[] DOScheme = null, float accuracyChangeLimit = -1, bool logging = true, string logFileName = "Train log.txt") //берем кол-во эпох, итераций в эпохе, данные для обучения, ответы
        {
            bool DOSchemeIsCorrect = NeuralNet.CheckDOScheme(DOScheme);

            if (DOScheme != null)
            {
                if (DOScheme.Length + 2 != NeuralNet.Outputs.Count)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Incorrect dropout scheme");
                    Console.ForegroundColor = ConsoleColor.White;
                }
            }

            List<List<float>> deltas = new List<List<float>>(); // создаем листы для хранения дельт

            for (int i = 0; i < NeuralNet.Outputs.Count; i++)
            {
                deltas.Add(new List<float>()); // добавляем ячейки для вычислений, так же как в листы output

                for (int k = 0; k < NeuralNet.Outputs[i].Count; k++)
                {
                    deltas[i].Add(0f);
                }
            }

            StreamWriter logger = new StreamWriter(logFileName, false);
            float prevTestMSE = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int iter = 0; iter < iterCount; iter++)
                {
                    for (int learnDataCounter = 0; learnDataCounter < learnData.Length; learnDataCounter++)
                    {
                        float[] NNOut;

                        if (DOScheme != null && DOSchemeIsCorrect == true)
                        {
                            NNOut = NeuralNet.RunDropOut(learnData[learnDataCounter], DOScheme); // получаем выходы нс 
                        }
                        else
                        {
                            NNOut = NeuralNet.Run(learnData[learnDataCounter]); // получаем выходы нс 
                        }

                        for (int i = 0; i < NNOut.Length; i++)
                        {
                            deltas[deltas.Count - 1][i] = (learnAnswers[learnDataCounter][i] - NNOut[i]) * NeuralNet.Descriptor.GetDerivedActivation()(NNOut[i]);//параллельно считаем дельты выходных нейронов
                        }

                        for (int layer = NeuralNet.Weights.Count - 1; layer >= 0; layer--)
                        {
                            for (int neuron = 0; neuron < NeuralNet.Outputs[layer].Count; neuron++)
                            {
                                for (int synapse = 0; synapse < NeuralNet.Outputs[layer + 1].Count; synapse++)
                                {
                                    deltas[layer][neuron] = deltas[layer + 1][synapse] * NeuralNet.Weights[layer][synapse][neuron]; // суммируем
                                }

                                deltas[layer][neuron] *= NeuralNet.Descriptor.GetDerivedActivation()(NeuralNet.Outputs[layer][neuron]); // домножаем на производную
                            }
                        }

                        for (int layer = NeuralNet.Weights.Count - 1; layer >= 0; layer--)
                        {
                            for (int neuron = 0; neuron < NeuralNet.Outputs[layer].Count; neuron++)
                            {
                                for (int synapse = 0; synapse < NeuralNet.Outputs[layer + 1].Count; synapse++)
                                {
                                    float deltaW = NeuralNet.Outputs[layer][neuron] * deltas[layer + 1][synapse];
                                    deltaW = learningRate * deltaW + moment * NeuralNet.WeightsDeltas[layer][synapse][neuron]; // тут сложно см ниже

                                    NeuralNet.Weights[layer][synapse][neuron] += deltaW;
                                    NeuralNet.WeightsDeltas[layer][synapse][neuron] = deltaW;
                                }
                            }
                        }

                        Console.WriteLine($"DataCounter = {learnDataCounter}");
                        /*
                        суть подсчета дельт такая:
                        т.к. у нас слоев весов 2(см. 1.jpg), а выходов и дельт всегда на 1 слой больше, так еще и веса у нас на вход, то
                        получается не очень удобно считать дельты. Поэтому я в перебираю не синапсы от нейрона, а нейроны, а нейроны от нейрона(2.jpg)
                        Как хранятся веса: Слой, нейрон, входы в этот нейрон
                         */
                    }

                    if (iter % 10 == 0)
                    {
                        float TotalMse = 0;

                        for (int testDataCounter = 0; testDataCounter < testData.Length; testDataCounter++)
                        {
                            float[] NNOut = NeuralNet.Run(testData[testDataCounter]); // получаем выходы нс 

                            float MSE = 0;

                            for (int i = 0; i < NNOut.Length; i++)
                            {
                                MSE += (float)Math.Pow(testAnswers[testDataCounter][i] - NNOut[i], 2); //подсчитываем сумму ошибок
                            }

                            MSE = MSE / NNOut.Length; // делим
                            TotalMse += MSE;
                        }

                        TotalMse /= testData.Length;

                        if (logging)
                        {
                            logger.WriteLine(TotalMse.ToString());

                            Console.WriteLine($"Epoch: {epoch + 1}\t Iteration: {iter}\t Avg test error: {TotalMse}");
                        }

                        if (Math.Abs(prevTestMSE - TotalMse) < accuracyChangeLimit)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.WriteLine("Accuracy limit!");
                            Console.ForegroundColor = ConsoleColor.White;
                            return 2;
                        }

                        prevTestMSE = TotalMse;
                    }
                }
            }

            logger.Close();

            return 1;
        }

        public void TrainResilentBackPropogation(int epochs, int iterCount, float[][] learnData, float[][] testData, float[][] learnAnswers, float[][] testAnswers, float etaPlus = 1.2f, float etaMinus = 0.5f, float deltaMax = 50.0f, float deltaMin = 1.0E-6f)
        {

        }

        /*
         1 - Обучение успешно закончено
         2 - Обучение прервано из-за достигнутого лимита
         3 - Остановка обучения по флагу
        -2 - Появился NaN
        */
    }
}
