using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.DataSets
{
    public class DataSetUnit
    {
        public float[] Data;
        public int Label;

        public DataSetUnit(float[] Data, int Label)
        {
            this.Data = Data;
            this.Label = Label;
        }
    }
}
