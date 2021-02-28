using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNet.Base;

namespace NeuralNet.Genetic
{
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

            Population[22].Rate = 1111;
            Population[2].Rate = 54564456;
        }

        public PopulationUnit GetPopulationUnit(int index)
        {
            return Population[index];
        }

        public void CreateNewPopulation(float mutationChance, int mutationPower, float crossoverChance, bool findMin = false)
        {
            PopulationUnit best1;
            PopulationUnit best2;

            Array.Sort(Population);

            if (!findMin)
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
                            if (cross < crossoverChance)
                            {
                                if (crossType < 0.5f)
                                {
                                    if (Base.Random.Next(0, 2) == 0)
                                    {
                                        Crossover.NN.Weights[k][j][l] = best1.NN.Weights[k][j][l];
                                    }
                                    else
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

            Population[0] = best1;
            Population[1] = best2;
        }
    }
}
