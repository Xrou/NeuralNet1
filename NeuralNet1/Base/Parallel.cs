using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNet.Base
{
    public static class Parallel
    {
        public static void ParallelFor(int from, int to, Action<int> body)
        {
            int numProcs = Environment.ProcessorCount;
            // количество оставшихся
            int remainingWorkItems = numProcs;
            int nextIteration = from;

            using (ManualResetEvent mre = new ManualResetEvent(false))
            {
                // создаём задания
                for (int p = 0; p < numProcs; p++)
                {
                    ThreadPool.QueueUserWorkItem(delegate
                    {
                        int index;
                        // отбираем по одному элементу на выполнение
                        while ((index = Interlocked.Increment(ref nextIteration) - 1) < to)
                        {
                            body(index);
                        }
                        if (Interlocked.Decrement(ref remainingWorkItems) == 0)
                            mre.Set();
                    });
                }
                // ждём, пока отработают все задания
                mre.WaitOne();
            }
        }

    }
}
