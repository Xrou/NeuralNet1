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

        public Net(int[] layersData) // layersData - кол-во нейронов в слое, кол-во элементов layersdata = кол-во слоев
        {
            Outputs.Add(new List<float>());

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
                        Weights[Weights.Count - 1][neuron].Add(0.5f);
                        WeightsDeltas[WeightsDeltas.Count - 1][neuron].Add(0.5f);
                    }
                }
            }
        }

        public void Train(int epochs, int iterCount, float LearningRate, float Moment, float[][] learnData, float[][] testData, float[][] learnAnswers, float[][] testAnswers) //берем кол-во эпох, итераций в эпохе, данные для обучения, ответы
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int iter = 0; iter < iterCount; iter++)
                {
                    for (int learnDataCounter = 0; learnDataCounter < learnData.Length; learnDataCounter++)
                    {

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

                }
            }

            return new float[0];
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
