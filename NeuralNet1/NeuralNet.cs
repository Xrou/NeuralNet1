using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Net
    {
        private const float E = 2.7182818285f;

        private int OutputsCount;

        // веса на вход нейрона
        private List<List<List<float>>> Weights = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи
        private List<List<List<float>>> WeightsDeltas = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи

        private List<List<float>> Outputs = new List<List<float>>(); // слой, номер нейрона в слое

        public Net(int[] layersData) // layersData - номер слоя, кол-во нейронов в слое
        {
            for (int layer = 1; layer < layersData.Length; layer++) // перебираем слои, пропускаем входной слой
            {
                Weights.Add(new List<List<float>>());
                WeightsDeltas.Add(new List<List<float>>());

                Outputs.Add(new List<float>());

                for (int neuron = 0; neuron < layersData[layer]; neuron++)
                {
                    Weights[layer - 1].Add(new List<float>()); // добавляем нейрон
                    WeightsDeltas[layer - 1].Add(new List<float>()); // добавляем нейрон

                    Outputs[layer - 1].Add(0f);

                    for (int inputsNeuron = 0; inputsNeuron < layersData[layer - 1]; inputsNeuron++)
                    {
                        Weights[layer - 1][neuron].Add(0.5f); // добавляем веса нейрону
                        WeightsDeltas[layer - 1][neuron].Add(0.5f); // добавляем дельты веса нейрону
                    }
                }
            }

            OutputsCount = layersData[layersData.Length - 1];
        }

        public void Train(int epochs, int iterCount, float[][] learnData, float[][] testData, float[][] learnAnswers, float[][] testAnswers) //берем кол-во эпох, итераций в эпохе, данные для обучения, ответы
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int iter = 0; iter < iterCount; iter++)
                {
                    for (int learnDataCounter = 0; learnDataCounter < learnData.Length; learnDataCounter++)
                    {
                        float[] output = Run(learnData[learnDataCounter]);

                        #region print
                        foreach (float val in output)
                        {
                            Console.Write(val + "\t");
                        }
                        #endregion

                        float Error = 0; //MSE error

                        #region MSE calc
                        for (int i = 0; i < OutputsCount; i++)
                        {
                            Error = output[i] - learnAnswers[learnDataCounter][i];
                        }

                        Error /= OutputsCount;
                        #endregion

                        float[] deltaOutput = new float[OutputsCount];

                        #region Deltas outputs calc
                        for (int i = 0; i < OutputsCount; i++)
                        {
                            deltaOutput[i] = (learnAnswers[learnDataCounter][i] - output[i]) * DerivedActivation(output[i]);
                        }
                        #endregion
                    }

                    if (iter % 10 == 0 && iter != 0)
                    {
                        //тест

                        Console.WriteLine($"Epoch: {epoch + 1} Iteration: {iter} Test accuracy: acc");
                    }
                }
            }
        }

        public float[] Run(float[] inputs)
        {
            for (int i = 0; i < Outputs.Count; i++)
            {
                for (int k = 0; k < Outputs[i].Count; k++) // чистим от предыдущих подсчетов
                {
                    Outputs[i][k] = 0;
                }
            }

            for (int i = 0; i < Weights[0].Count; i++)
            {
                for (int k = 0; k < Weights[0][i].Count; k++)
                {
                    Outputs[0][i] += inputs[k] * Weights[0][i][k];
                }

                Outputs[0][i] = Activation(Outputs[0][i]);
            }

            for (int i = 1; i < Weights.Count; i++)
            {
                for (int k = 0; k < Weights[i].Count; k++)
                {
                    for (int j = 0; j < Weights[i - 1].Count; j++)
                    {
                        Outputs[i][k] += Outputs[i - 1][j] * Weights[i][k][j];
                    }

                    Outputs[i][k] = Activation(Outputs[i][k]);
                }
            }

            return Outputs[Outputs.Count - 1].ToArray(); //возвращаем выходы из сети
        }

        private float Activation(float x)
        {
            return (float)(1 / (1 + Math.Pow(E, -x)));
        }

        private float DerivedActivation(float x)
        {
            return (1 - x) * x;
        }
    }
}
