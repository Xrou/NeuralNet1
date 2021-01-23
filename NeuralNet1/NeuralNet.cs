using System;
using System.Xml;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.IO;

namespace NeuralNet
{
    public class FeedForwardNN
    {
        private Random random;
        public delegate float Activation(float x);
        public delegate float DerivedActivation(float x);

        Activation activation;
        DerivedActivation derivedActivation;

        // веса на вход нейрона
        [System.Xml.Serialization.XmlElement("Weights")]
        private List<List<List<float>>> Weights = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи
        private List<List<List<float>>> WeightsDeltas = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи

        private List<List<float>> Outputs = new List<List<float>>(); // слой, номер нейрона в слое


        public FeedForwardNN(int[] layersData, Activation activation, DerivedActivation derivedActivation) // layersData - кол-во нейронов в слое, кол-во элементов layersdata = кол-во слоев
        {
            random = new Random();
            Outputs.Add(new List<float>());

            this.activation = activation;
            this.derivedActivation = derivedActivation;

            for (int i = 0; i < layersData[0]; i++) //добавляем входы как выходы
            {
                Outputs[0].Add(0);
            }

            for (int layer = 1; layer < layersData.Length; layer++)
            {
                Weights.Add(new List<List<float>>());
                WeightsDeltas.Add(new List<List<float>>());

                Outputs.Add(new List<float>());

                for (int neuron = 0; neuron < layersData[layer]; neuron++)
                {
                    Weights[Weights.Count - 1].Add(new List<float>());
                    WeightsDeltas[Weights.Count - 1].Add(new List<float>());

                    Outputs[Outputs.Count - 1].Add(0f);

                    for (int synapse = 0; synapse < layersData[layer - 1]; synapse++)
                    {
                        Weights[Weights.Count - 1][neuron].Add(random.Next(-1, 1) + (float)random.NextDouble());
                        WeightsDeltas[WeightsDeltas.Count - 1][neuron].Add(0f);
                    }
                }
            }
        }

        public void Train(int epochs, int iterCount, float LearningRate, float Moment, float[][] learnData, float[][] testData, float[][] learnAnswers, float[][] testAnswers, float[] DOScheme = null, float accuracyChangeLimit = -1, bool logging = true, string logFileName = "Train log.txt") //берем кол-во эпох, итераций в эпохе, данные для обучения, ответы
        {
            if (DOScheme != null)
            {
                if (DOScheme.Length + 2 != Outputs.Count)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Incorrect dropout scheme");
                    Console.WriteLine($"Need {Outputs.Count - 2} elements");
                    Console.ForegroundColor = ConsoleColor.White;
                }
            }

            List<List<float>> deltas = new List<List<float>>(); // создаем листы для хранения дельт

            for (int i = 0; i < Outputs.Count; i++)
            {
                deltas.Add(new List<float>()); // добавляем ячейки для вычислений, так же как в листы output

                for (int k = 0; k < Outputs[i].Count; k++)
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

                        if (DOScheme != null)
                        {
                            NNOut = RunDropOut(learnData[learnDataCounter], DOScheme); // получаем выходы нс 
                        }
                        else
                        {
                            NNOut = Run(learnData[learnDataCounter]); // получаем выходы нс 
                        }

                        for (int i = 0; i < NNOut.Length; i++)
                        {
                            deltas[deltas.Count - 1][i] = (learnAnswers[learnDataCounter][i] - NNOut[i]) * derivedActivation(NNOut[i]);//параллельно считаем дельты выходных нейронов
                        }

                        for (int layer = Weights.Count - 1; layer >= 0; layer--)
                        {
                            for (int neuron = 0; neuron < Outputs[layer].Count; neuron++)
                            {
                                for (int synapse = 0; synapse < Outputs[layer + 1].Count; synapse++)
                                {
                                    deltas[layer][neuron] = deltas[layer + 1][synapse] * Weights[layer][synapse][neuron]; // суммируем
                                }

                                deltas[layer][neuron] *= derivedActivation(Outputs[layer][neuron]); // домножаем на производную
                            }
                        }

                        for (int layer = Weights.Count - 1; layer >= 0; layer--)
                        {
                            for (int neuron = 0; neuron < Outputs[layer].Count; neuron++)
                            {
                                for (int synapse = 0; synapse < Outputs[layer + 1].Count; synapse++)
                                {
                                    float deltaW = Outputs[layer][neuron] * deltas[layer + 1][synapse];
                                    deltaW = LearningRate * deltaW + Moment * WeightsDeltas[layer][synapse][neuron]; // тут сложно см ниже

                                    Weights[layer][synapse][neuron] += deltaW;
                                    WeightsDeltas[layer][synapse][neuron] = deltaW;
                                }
                            }
                        }

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
                            float[] NNOut = Run(testData[testDataCounter]); // получаем выходы нс 

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
                            return;
                        }

                        prevTestMSE = TotalMse;
                    }
                }
            }

            logger.Close();
        }

        public float[] Run(float[] inputs)
        {
            if (inputs.Length != Outputs[0].Count) //сравниваем кол-во нейронов на входе и кол-во входов
            {
                Console.WriteLine($"Run: неправильное количество входов. Необходимо: {Weights[0].Count}, получено: {inputs.Length}");
                return null;
            }

            for (int i = 1; i < Outputs.Count; i++) // чистим выходы от предыдущих данных
            {
                for (int k = 0; k < Outputs[i].Count; k++)
                {
                    Outputs[i][k] = 0;
                }
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                Outputs[0][i] = inputs[i]; //присваиваем входам значения входов
            }

            for (int layer = 0; layer < Weights.Count; layer++) // перебор слоёв
            {
                for (int neuron = 0; neuron < Weights[layer].Count; neuron++)
                {
                    for (int synapse = 0; synapse < Weights[layer][neuron].Count; synapse++)
                    {
                        Outputs[layer + 1][neuron] += Outputs[layer][synapse] * Weights[layer][neuron][synapse];
                    }

                    Outputs[layer + 1][neuron] = activation(Outputs[layer + 1][neuron]);
                }
            }

            return Outputs[Outputs.Count - 1].ToArray();
        }

        public float[] RunDropOut(float[] inputs, float[] DropOutScheme)
        {
            if (inputs.Length != Outputs[0].Count) //сравниваем кол-во нейронов на входе и кол-во входов
            {
                Console.WriteLine($"Run: неправильное количество входов. Необходимо: {Weights[0].Count}, получено: {inputs.Length}");
                return null;
            }

            for (int i = 1; i < Outputs.Count; i++) // чистим выходы от предыдущих данных
            {
                for (int k = 0; k < Outputs[i].Count; k++)
                {
                    Outputs[i][k] = 0;
                }
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                Outputs[0][i] = inputs[i]; //присваиваем входам значения входов
            }

            for (int layer = 0; layer < Weights.Count; layer++) // перебор слоёв
            {
                for (int neuron = 0; neuron < Weights[layer].Count; neuron++)
                {
                    for (int synapse = 0; synapse < Weights[layer][neuron].Count; synapse++)
                    {
                        Outputs[layer + 1][neuron] += Outputs[layer][synapse] * Weights[layer][neuron][synapse];
                    }

                    Outputs[layer + 1][neuron] = activation(Outputs[layer + 1][neuron]);
                }
            }

            for (int i = 0; i < DropOutScheme.Length; i++)
            {
                for (int k = 0; k < Outputs[i + 1].Count; k++)
                {
                    if (random.NextDouble() < DropOutScheme[i])
                    {
                        Outputs[i + 1][k] = 0;
                    }
                }
            }

            return Outputs[Outputs.Count - 1].ToArray();
        }

        public void SaveWeights(string fn)
        {
            XmlSerializer xmlSerializer = new XmlSerializer(typeof(List<List<List<float>>>));

            using (FileStream fs = new FileStream(fn, FileMode.Create))
            {
                xmlSerializer.Serialize(fs, Weights);
            }
        }

        public void ReadWeights(string fn)
        {
            XmlSerializer xmlSerializer = new XmlSerializer(typeof(List<List<List<float>>>));

            using (FileStream fs = new FileStream(fn, FileMode.Open))
            {
                Weights = (List<List<List<float>>>)xmlSerializer.Deserialize(fs);
            }

        }
    }
}
