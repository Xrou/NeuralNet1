using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NeuralNet.DataSets
{
    public class MnistDataSet : DataSet
    {
        public MnistDataSet(string Path, int BatchSize, int ElementsCount)
            : base(Path, BatchSize, ElementsCount) { }

        public override DataSetUnit[] GetBatch()
        {
            List<DataSetUnit> units = new List<DataSetUnit>();

            if ((LastItemIndex + BatchSize) >= ElementsCount)
            {
                for (int i = LastItemIndex; i < ElementsCount; i++)
                {
                    units.Add(GetUnit(LastItemIndex + i));
                }

                LastItemIndex += BatchSize;
                LastItemIndex -= ElementsCount;
            }
            else
            {
                for (int i = LastItemIndex; i < LastItemIndex + BatchSize; i++)
                {
                    units.Add(GetUnit(LastItemIndex + i));
                }

                LastItemIndex += BatchSize;
            }

            return units.ToArray();
        }

        private DataSetUnit GetUnit(int index)
        {
            string imageIndex = "0" + index;
            int len = imageIndex.Length;

            for (int i = 0; i < 6 - len; i++)
            {
                imageIndex = "0" + imageIndex;
            }

            string imagePath = "";

            for (int i = 0; i < 10; i++)
            {
                FileInfo fi = new FileInfo(Path + $"{imageIndex}-num{i}.png");

                if (fi.Exists)
                {
                    imagePath = fi.FullName;
                    break;
                }
            }

            Bitmap bitmap = new Bitmap(imagePath);
            List<float> data = new List<float>();

            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);

            BitmapData bmpData = bitmap.LockBits(rect, ImageLockMode.ReadWrite, bitmap.PixelFormat);

            IntPtr ptr = bmpData.Scan0;

            int bytes = Math.Abs(bmpData.Stride) * bitmap.Height;
            byte[] rgbValues = new byte[bytes];
            Marshal.Copy(ptr, rgbValues, 0, bytes);

            for (int counter = 0; counter < rgbValues.Length; counter += 4)
            {
                float gray = (rgbValues[counter] + rgbValues[counter + 1] + rgbValues[counter + 2]) / 3;
                gray = Base.Normalize.Minimax(gray, 0, 255);
                data.Add(gray);
            }

            Marshal.Copy(rgbValues, 0, ptr, bytes);
            bitmap.UnlockBits(bmpData);

            int label = Convert.ToInt32(imagePath[imagePath.Length - 5].ToString());

            return new DataSetUnit(data.ToArray(), label);
        }
    }
}
