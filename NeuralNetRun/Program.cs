using System;
using System.Windows.Forms;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using NeuralNet;


namespace NeuralNetRun
{
    class Program
    {
        static Form form;

        static void Main(string[] args)
        {
            Random rnd = new Random();

            float[][] learningData = new float[200][];
            float[][] learningAnswers = new float[200][];

            float[][] testData = new float[20][];
            float[][] testAnswers = new float[20][];

            float lrMin = 100;
            float lrMax = 0;

            for (int i = 0; i < learningData.Length; i++)
            {
                float var = rnd.Next(10, 81);

                learningData[i] = new float[] { var };
                if (learningData[i][0] <= 40) learningAnswers[i] = new float[] { 0 };
                if (learningData[i][0] > 40) learningAnswers[i] = new float[] { 1 };

                if (learningData[i][0] < lrMin) lrMin = learningData[i][0];
                if (learningData[i][0] > lrMax) lrMax = learningData[i][0];
            }

            for (int i = 0; i < testData.Length; i++)
            {
                float var = rnd.Next(10, 81);

                testData[i] = new float[] { var };
                if (testData[i][0] <= 40) testAnswers[i] = new float[] { 0 };
                if (testData[i][0] > 40) testAnswers[i] = new float[] { 1 };
            }

            for (int i = 0; i < learningData.Length; i++)
            {
                learningData[i][0] = Normalize.Minimax(learningData[i][0], lrMin, lrMax);
            }

            for (int i = 0; i < testData.Length; i++)
            {
                testData[i][0] = Normalize.Minimax(testData[i][0], lrMin, lrMax);
            }

            FeedForwardNN nn = new FeedForwardNN(new int[] { 1, 3, 3, 1 }, Activation.Sigmoid, Activation.DerivedSigmoid);

            nn.Train(10, 1000, 0.1f, 0.3f,
                learningData, testData,
                learningAnswers, testAnswers);

            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(39, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(40, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(41, lrMin, lrMax) })[0]);
            Console.WriteLine(nn.Run(new float[] { Normalize.Minimax(42, lrMin, lrMax) })[0]);

            form = new Form();
            form.Size = new Size(1050, 550);
            form.Paint += Form_Paint;
            form.ShowDialog();
        }

        static Graphics grBmp;
        static Pen MyPen;
        static Bitmap bmp;

        private static void Form_Paint(object sender, PaintEventArgs e)
        {
            bmp = new Bitmap(1000, 500);
            grBmp = Graphics.FromImage(bmp);
            MyPen = new Pen(Color.Black);

            Graphics graph = e.Graphics;
            grBmp.Clear(Color.White);
            MyPen.Color = Color.Black;

            List<int> YCoord = new List<int>();

            using (StreamReader sr = new StreamReader("./Train log.txt", Encoding.Default))
            {
                string line;

                while ((line = sr.ReadLine()) != null)
                {
                    int val = Convert.ToInt32(Convert.ToSingle(line) * 10000);
                    YCoord.Add(1000 - val);
                }
            }

            for (int i = 0; i < YCoord.Count - 2; i++)
            {
                grBmp.DrawLine(MyPen, new Point(i * 2, YCoord[i]/2), new Point((i + 1) * 2, YCoord[i + 1]/2));
            }

            graph.DrawImage(bmp, 0, 0);
        }
    }
}
