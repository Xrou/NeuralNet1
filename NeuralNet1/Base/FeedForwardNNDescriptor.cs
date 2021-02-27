using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Base
{
    public class FeedForwardNNDescriptor
    {
        public int[] LayersData { get; }
        public Activation Activation { get; }
        public DerivedActivation DerivedActivation { get; }

        public FeedForwardNNDescriptor(int[] LayersData, Activation Activation, DerivedActivation DerivedActivation)
        {
            this.LayersData = LayersData;
            this.Activation = Activation;
            this.DerivedActivation = DerivedActivation;
        }
    }
}
