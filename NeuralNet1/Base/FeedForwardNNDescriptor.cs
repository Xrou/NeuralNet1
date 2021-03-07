using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Base
{
    [Serializable]
    public class FeedForwardNNDescriptor
    {
        private string activationStr;

        public int[] LayersData;
        public string Activation
        {
            get { return activationStr; }
            set
            {
                activationStr = value;
                var act = Activations.FromString(activationStr); 
                activation = act.Item1;
                derivedActivation = act.Item2;
            }
        }
        private Activation activation;
        private DerivedActivation derivedActivation;

        public FeedForwardNNDescriptor() { }

        public FeedForwardNNDescriptor(int[] LayersData, Activation Activation, DerivedActivation DerivedActivation)
        {
            this.LayersData = LayersData;
            this.activation = Activation;
            this.derivedActivation = DerivedActivation;

            this.Activation = Activations.AsString(Activation);
        }

        public Activation GetActivation()
        {
            return activation;
        }

        public DerivedActivation GetDerivedActivation()
        {
            return derivedActivation;
        }
    }
}
