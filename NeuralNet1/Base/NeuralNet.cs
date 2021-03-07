using System;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;

namespace NeuralNet.Base
{
    public delegate float Activation(float x);
    public delegate float DerivedActivation(float x);

    public class FeedForwardNN : ICloneable
    {
        public FeedForwardNNDescriptor Descriptor;

        [System.Xml.Serialization.XmlElement("Weights")]
        public List<List<List<float>>> Weights = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи
        public List<List<List<float>>> WeightsDeltas = new List<List<List<float>>>(); // слой, номер нейрона в слое, номер связи

        public List<List<float>> Outputs = new List<List<float>>(); // слой, номер нейрона в слое

        public FeedForwardNN(FeedForwardNNDescriptor descriptor) // layersData - кол-во нейронов в слое, кол-во элементов layersdata = кол-во слоев
        {
            if (descriptor.LayersData != null)
            {
                Descriptor = descriptor;
                InitNN(descriptor.LayersData);
            }
        }

        private void InitNN(int[] layersData)
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
                        Weights[Weights.Count - 1][neuron].Add(Random.NextFloat(-1, 1));

                        WeightsDeltas[WeightsDeltas.Count - 1][neuron].Add(0f);
                    }
                }
            }
        }

        public bool CheckDOScheme(float[] DOScheme)
        {
            if (DOScheme != null)
            {
                if (DOScheme.Length + 2 != Outputs.Count)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Incorrect dropout scheme");
                    Console.WriteLine($"Need {Outputs.Count - 2} elements");
                    Console.ForegroundColor = ConsoleColor.White;
                    return false;
                }
            }

            return true;
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

                    Outputs[layer + 1][neuron] = Descriptor.GetActivation()(Outputs[layer + 1][neuron]);
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

                    Outputs[layer + 1][neuron] = Descriptor.GetActivation()(Outputs[layer + 1][neuron]);
                }
            }

            for (int i = 0; i < DropOutScheme.Length; i++)
            {
                for (int k = 0; k < Outputs[i + 1].Count; k++)
                {
                    if (Random.NextFloat(0, 1) < DropOutScheme[i])
                    {
                        Outputs[i + 1][k] = 0;
                    }
                }
            }

            return Outputs[Outputs.Count - 1].ToArray();
        }

        public void SaveWeights(string fn)
        {
            string fn1;
            string fn2;

            if (fn.EndsWith(".xml"))
            {
                fn1 = fn.Substring(0, fn.Length - 4);
                fn1 += " w.xml";

                fn2 = fn.Substring(0, fn.Length - 4);
                fn2 += " d.xml";

            }
            else
            {
                fn1 = fn + " w.xml";
                fn2 = fn + " d.xml";
            }

            XmlSerializer xmlSerializer = new XmlSerializer(typeof(List<List<List<float>>>));

            using (FileStream fs = new FileStream(fn1, FileMode.Create))
            {
                xmlSerializer.Serialize(fs, Weights);
            }

            xmlSerializer = new XmlSerializer(typeof(FeedForwardNNDescriptor));

            using (FileStream fs = new FileStream(fn2, FileMode.Create))
            {
                xmlSerializer.Serialize(fs, Descriptor);
            }
        }

        public void ReadWeights(string fn)
        {
            string fn1;
            string fn2;

            if (fn.EndsWith(".xml"))
            {
                fn1 = fn.Substring(0, fn.Length - 4);
                fn1 += " w.xml";

                fn2 = fn.Substring(0, fn.Length - 4);
                fn2 += " d.xml";
            }
            else
            {
                fn1 = fn + " w.xml";
                fn2 = fn + " d.xml";
            }

            XmlSerializer xmlSerializer = new XmlSerializer(typeof(FeedForwardNNDescriptor));

            using (FileStream fs = new FileStream(fn2, FileMode.Open))
            {
                Descriptor = (FeedForwardNNDescriptor)xmlSerializer.Deserialize(fs);
            }

            InitNN(Descriptor.LayersData);

            xmlSerializer = new XmlSerializer(typeof(List<List<List<float>>>));

            using (FileStream fs = new FileStream(fn1, FileMode.Open))
            {
                Weights = (List<List<List<float>>>)xmlSerializer.Deserialize(fs);
            }
        }

        public object Clone()
        {
            var FFNN = new FeedForwardNN(Descriptor);

            FFNN.Weights = Weights;
            FFNN.WeightsDeltas = WeightsDeltas;
            FFNN.Outputs = Outputs;

            return FFNN;
        }
    }
}
