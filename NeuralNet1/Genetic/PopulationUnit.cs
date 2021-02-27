using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet.Base;

namespace NeuralNet.Genetic
{
    public class PopulationUnit : ICloneable
    {
        public FeedForwardNN NN;
        public float Rate = 0;

        public float[] Outputs = new float[0];

        public PopulationUnit(FeedForwardNN NN)
        {
            this.NN = NN;

            Outputs = new float[NN.Descriptor.LayersData[NN.Descriptor.LayersData.Length - 1]];
        }

        public float[] GetNNOutput(float[] inputs)
        {
            Outputs = NN.Run(inputs);

            return Outputs;
        }

        public object Clone()
        {
            var PU =  new PopulationUnit(NN);

            PU.NN = NN;
            PU.Outputs = Outputs;
            PU.Rate = Rate;

            return PU;
        }
    }
}
