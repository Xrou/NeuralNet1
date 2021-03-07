using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.DataSets
{
    public abstract class DataSet
    {
        public string Path;
        public int BatchSize;

        protected int LastItemIndex;
        protected int ElementsCount;

        public DataSet(string Path, int BatchSize, int ElementsCount)
        {
            this.Path = Path;
            this.BatchSize = BatchSize;
            this.ElementsCount = ElementsCount;
            this.LastItemIndex = 0;
        }

        public abstract DataSetUnit[] GetBatch();
    }
}
