using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet;

namespace NeuralNetRun
{
    class Program
    {
        static void Main(string[] args)
        {
            Net NeuralNet = new Net(new int[] { 2, 3, 1 });//предсказываем "или" и "исключающее или"

            foreach (float val in NeuralNet.Run(new float[] { 1, 1 }))
            {
                Console.Write(val + "\t");
            }

            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 1, 0 }))
            {
                Console.Write(val + "\t");
            }

            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 0, 1 }))
            {
                Console.Write(val + "\t");
            }

            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 0, 0 }))
            {
                Console.Write(val + "\t");
            }

            Console.WriteLine();
            Console.WriteLine();

            NeuralNet.Train(1, 1, 0.001f, 0.3f,
                new float[][] { new float[] { 1, 0 } }, //данные для обучения
                new float[][] { new float[] { 1, 1 } }, //данные для тренировки
                new float[][] { new float[] { 1 } }, //ответы для обучения
                new float[][] { new float[] { 0 } });//ответы для тестов

            Console.ReadLine();
        }
    }
}
