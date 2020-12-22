using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NeuralNet;

namespace NeuralNetRun
{
    class Program
    {
        static void Main(string[] args)
        {
            FeedForwardNN NeuralNet = new FeedForwardNN(new int[] { 2, 5, 3, 2 }, Activation.Sigmoid, Activation.DerivedSigmoid);//предсказываем "или" и "и"
            bool weightsLoaded = false;
            Console.WriteLine("Read prev weights?(y, n)");

            if (Console.ReadLine() == "y")
            {
                weightsLoaded = true;
                NeuralNet.ReadWeights("Weights.xml");

                Thread thread1 = new Thread(new ThreadStart(calc));

                thread1.Priority = ThreadPriority.Highest;

                thread1.Start();
            }
            else
            {
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

                Console.WriteLine("\n^ ANSWERS BEFORE TRAIN");
                Console.WriteLine("|");

                Stopwatch stopwatch = new Stopwatch();

                stopwatch.Start();

                NeuralNet.Train(130, 1000, 0.01f, 0.3f,
                    new float[][] {
                    new float[] { 1, 1 },
                    new float[] { 0, 1 },
                    new float[] { 1, 0 },
                    new float[] { 0, 0 }}, //данные для обучения

                    new float[][] {
                    new float[] { 1, 0 },
                    new float[] { 0, 1 },
                    new float[] { 0, 0 },
                    new float[] { 1, 1 }}, //данные для тренировки

                    new float[][] {
                    new float[] { 1, 1 },
                    new float[] { 1, 0 },
                    new float[] { 1, 0 },
                    new float[] { 0, 0 } }, //ответы для обучения

                    new float[][] {
                    new float[] { 1, 0 },
                    new float[] { 1, 0 },
                    new float[] { 0, 0 },
                    new float[] { 1, 1 } }, //ответы для тестов

                    new float[] { 0.25f, 0.1f }, logging:true);//схема дропаута 

                stopwatch.Stop();
                Console.WriteLine($"Elapsed time:{stopwatch.ElapsedMilliseconds}");

                Console.WriteLine("| ANSWERS AFTER TRAIN");
                Console.WriteLine("\\/");
            }

            foreach (float val in NeuralNet.Run(new float[] { 0, 0 }))
            {
                Console.Write(val + "\t");
            }
            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 0, 1 }))
            {
                Console.Write(val + "\t");
            }
            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 1, 0 }))
            {
                Console.Write(val + "\t");
            }
            Console.WriteLine();

            foreach (float val in NeuralNet.Run(new float[] { 1, 1 }))
            {
                Console.Write(val + "\t");
            }

            Console.WriteLine();

            if (!weightsLoaded)
            {
                Console.WriteLine("Save weights?(y, n)");

                if (Console.ReadLine() == "y")
                {
                    NeuralNet.SaveWeights("Weights.xml");
                }
            }

            Console.ReadLine();
        }

        static void calc()
        {
            double sum = 0;

            for (int i = 0; i < 999999999; i++)
            {
                sum += (i * i);
            }
        }
    }
}
