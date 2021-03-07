using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NeuralNet.Base;

namespace NeuralNet.Genetic
{
    public delegate void PostIterationEvolutionMethod(int iter, PopulationUnit best1);

    public class PopulationController
    {
        public int PopulationCount;

        private PopulationUnit[] Population;
        private FeedForwardNNDescriptor descriptor;

        public PopulationController(FeedForwardNNDescriptor descriptor, int PopulationCount)
        {
            if (PopulationCount < 2)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Population count must be more than 2");
                Console.ForegroundColor = ConsoleColor.White;
            }

            this.PopulationCount = PopulationCount;
            this.descriptor = descriptor;

            Population = new PopulationUnit[PopulationCount];

            for (int i = 0; i < PopulationCount; i++)
            {
                Population[i] = new PopulationUnit(new FeedForwardNN(descriptor));
            }
        }

        public PopulationController(FeedForwardNN nn, int PopulationCount)
        {
            if (PopulationCount < 2)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Population count must be more than 2");
                Console.ForegroundColor = ConsoleColor.White;
            }

            this.PopulationCount = PopulationCount;
            this.descriptor = nn.Descriptor;

            Population = new PopulationUnit[PopulationCount];

            for (int i = 0; i < PopulationCount; i++)
            {
                Population[i] = new PopulationUnit(nn);
            }
        }

        public PopulationUnit GetPopulationUnit(int index)
        {
            return Population[index];
        }

        public void StartEvolution(int iterations, float mutationChance, int mutationPower, float crossoverChance, float[][] testData, float[][] testAnswers, PostIterationEvolutionMethod postIterationEvolutionMethod, bool bestMin = false)
        {
            for (int iter = 0; iter < iterations; iter++)
            {
                SetRateMSE(testData, testAnswers);

                postIterationEvolutionMethod(iter, GetBest(bestMin:bestMin));

                CreateNewPopulation(mutationChance, mutationPower, crossoverChance, bestMin: true);
            }
        }

        public PopulationUnit GetBest(bool bestMin = false)
        {
            Array.Sort(Population);

            if (!bestMin)
            {
                return Population[PopulationCount - 1];
            }
            else
            {
                return Population[0];
            }
        }

        public (float, float) GetBestRate(bool bestMin = false)
        {
            Array.Sort(Population);

            if (!bestMin)
            {
                return (Population[PopulationCount - 1].Rate, Population[PopulationCount - 1].Rate);
            }
            else
            {
                return (Population[0].Rate, Population[1].Rate);
            }
        }

        public void SetRateMSE(float[][] inputs, float[][] answers)
        {
            Base.Parallel.ParallelFor(0, PopulationCount, (int i) => 
            {
                PopulationUnit unit = GetPopulationUnit(i);

                float TotalMse = 0;

                for (int testDataCounter = 0; testDataCounter < inputs.Length; testDataCounter++)
                {
                    float[] NNOut = unit.NNRun(inputs[testDataCounter]); // получаем выходы нс 

                    float MSE = 0;

                    for (int k = 0; k < NNOut.Length; k++)
                    {
                        MSE += (float)Math.Pow(answers[testDataCounter][k] - NNOut[k], 2); //подсчитываем сумму ошибок
                    }

                    MSE = MSE / NNOut.Length; // делим
                    TotalMse += MSE;
                }

                TotalMse /= answers.Length;

                unit.Rate = TotalMse;
            });
        }

        public void CreateNewPopulation(float mutationChance, int mutationPower, float crossoverChance, bool bestMin = false)
        {
            PopulationUnit best1;
            PopulationUnit best2;

            Array.Sort(Population);

            if (!bestMin)
            {
                best1 = (PopulationUnit)Population[PopulationCount - 1].Clone();
                best2 = (PopulationUnit)Population[PopulationCount - 2].Clone();
            }
            else
            {
                best1 = (PopulationUnit)Population[0].Clone();
                best2 = (PopulationUnit)Population[1].Clone();
            }

            for (int i = 2; i < PopulationCount; i++)
            {
                PopulationUnit Crossover = new PopulationUnit(new FeedForwardNN(descriptor));

                float cross = Base.Random.NextFloat(0, 1);
                float crossType = Base.Random.NextFloat(0, 1);

                int kLim = 0;

                if (crossType >= 0.5f && cross < crossoverChance)
                {
                    kLim = Convert.ToInt32(Base.Random.Next(0, best1.NN.Weights.Count - 1));
                }

                for (int k = 0; k < best1.NN.Weights.Count; k++)
                {
                    for (int j = 0; j < best1.NN.Weights[k].Count; j++)
                    {
                        for (int l = 0; l < best1.NN.Weights[k][j].Count; l++)
                        {
                            Crossover.NN.Weights[k][j][l] = best1.NN.Weights[k][j][l];

                            if (cross < crossoverChance)
                            {
                                if (crossType < 0.5f)
                                {
                                    if (Base.Random.Next(0, 2) == 0)
                                    {
                                        Crossover.NN.Weights[k][j][l] = best2.NN.Weights[k][j][l];
                                    }
                                }
                                else
                                {
                                    if (kLim > k)
                                    {
                                        Crossover.NN.Weights[k][j][l] = best2.NN.Weights[k][j][l];
                                    }
                                    else
                                    {
                                        Crossover.NN.Weights[k][j][l] = best1.NN.Weights[k][j][l];
                                    }
                                }
                            }

                            if (Base.Random.NextFloat(0, 1) < mutationChance)
                            {
                                Crossover.NN.Weights[k][j][l] += Base.Random.NextFloat(-mutationPower, mutationPower);
                            }
                        }
                    }
                }

                Population[i] = Crossover;
            }

            best1.Rate = 0;
            best2.Rate = 0;

            Population[0] = best1;
            Population[1] = best2;
        }
    }
}
