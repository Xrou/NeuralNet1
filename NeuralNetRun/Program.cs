﻿using System;
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
            Net NeuralNet = new Net(new int[] { 2, 3, 4, 2 });//предсказываем "или" и "исключающее или"

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

            NeuralNet.Train(1, 1,
                new float[][] { new float[] { 1, 0 } }, //данные для обучения
                new float[][] { new float[] { 1, 1 } }, //данные для тренировки
                new float[][] { new float[] { 1, 1 } }, //ответы для обучения
                new float[][] { new float[] { 0, 1 } });//ответы для тестов

            Console.ReadLine();
        }
    }
}
