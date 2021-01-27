﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetRun
{
    public class Data
    {
        public static float[][] learnDataNet = new float[][]
        {
            new float[] { 57,  38,  44,  55,  52 },
            new float[] { 42,  42,  35,  44,  35 },
            new float[] { 54,  53,  36,  55,  65 },
            new float[] { 62,  48,  54,  72,  68 },
            new float[] { 46,  38,  49,  50,  52 },
            new float[] { 62,  85,  58,  62,  90 },
            new float[] { 62,  42,  63,  44,  39 },
            new float[] { 65,  53,  54,  66,  65 },
            new float[] { 54,  40,  73,  50,  52 },
            new float[] { 42,  38,  35,  38,  39 },
            new float[] { 50,  42,  45,  33,  65 },
            new float[] { 57,  56,  40,  45,  35 },
            new float[] { 65,  38,  63,  62,  47 },
            new float[] { 57,  50,  45,  55,  47 },
            new float[] { 62,  66,  67,  62,  56 },
            new float[] { 43,  35,  40,  45,  44 },
            new float[] { 43,  32,  63,  45,  31 },
            new float[] { 46,  50,  30,  45,  44 },
            new float[] { 54,  43,  40,  72,  47 },
            new float[] { 43,  43,  58,  55,  44 },
            new float[] { 46,  46,  40,  38,  35 },
            new float[] { 43,  58,  40,  50,  53 },
            new float[] { 62,  43,  49,  62,  52 },
            new float[] { 28,  36,  44,  38,  31 },
            new float[] { 57,  46,  49,  55,  47 },
            new float[] { 50,  46,  54,  45,  47 },
            new float[] { 65,  65,  63,  61,  60 },
            new float[] { 50,  39,  68,  61,  52 },
            new float[] { 54,  42,  44,  33,  48 },
            new float[] { 57,  53,  35,  55,  43 },
            new float[] { 50,  42,  49,  50,  43 },
            new float[] { 43,  48,  58,  55,  48 },
            new float[] { 72,  71,  54,  73,  56 },
            new float[] { 43,  46,  45,  61,  52 },
            new float[] { 35,  42,  45,  50,  26 },
            new float[] { 43,  39,  40,  38,  35 },
            new float[] { 54,  42,  40,  55,  36 },
            new float[] { 43,  42,  49,  33,  39 },
            new float[] { 65,  56,  58,  38,  57 },
            new float[] { 69,  58,  40,  66,  52 },
            new float[] { 35,  68,  54,  50,  68 },
            new float[] { 62,  46,  67,  61,  68 },
            new float[] { 62,  46,  67,  73,  52 },
            new float[] { 50,  42,  25,  55,  44 },
            new float[] { 65,  46,  35,  38,  44 },
            new float[] { 57,  46,  67,  55,  48 },
            new float[] { 50,  58,  58,  50,  44 },
            new float[] { 54,  46,  40,  66,  60 },
            new float[] { 69,  68,  58,  66,  77 },
            new float[] { 54,  53,  40,  45,  57 },
            new float[] { 46,  46,  63,  38,  52 },
            new float[] { 54,  42,  54,  61,  57 },
            new float[] { 57,  56,  45,  50,  57 },
            new float[] { 54,  39,  35,  45,  48 },
            new float[] { 50,  42,  59,  61,  65 },
            new float[] { 62,  46,  49,  55,  60 },
            new float[] { 54,  46,  58,  45,  44 },
            new float[] { 57,  53,  73,  50,  68 },
            new float[] { 43,  39,  25,  45,  44 },
            new float[] { 57,  49,  63,  55,  65 },
            new float[] { 38,  49,  35,  61,  31 },
            new float[] { 50,  46,  54,  55,  39 },
            new float[] { 62,  42,  49,  72,  48 },
            new float[] { 69,  56,  54,  50,  73 },
            new float[] { 38,  49,  35,  38,  31 },
            new float[] { 54,  56,  54,  61,  52 },
            new float[] { 57,  42,  54,  72,  36 },
            new float[] { 38,  39,  35,  33,  26 },
            new float[] { 73,  49,  25,  55,  52 },
            new float[] { 54,  53,  49,  50,  60 },
            new float[] { 43,  46,  63,  52,  48 },
            new float[] { 50,  53,  45,  50,  52 },
            new float[] { 57,  53,  40,  55,  48 },
            new float[] { 43,  46,  54,  33,  48 },
            new float[] { 69,  49,  58,  55,  73 }
        };

        public static float[][] learnAnswersNet = new float[][]
        {
            new float[] { 43, 49 },
            new float[] { 48, 47 },
            new float[] { 48, 49 },
            new float[] { 58, 64 },
            new float[] { 43, 60 },
            new float[] { 63, 87 },
            new float[] { 43, 49 },
            new float[] { 40, 53 },
            new float[] { 40, 53 },
            new float[] { 36, 38 },
            new float[] { 40, 49 },
            new float[] { 47, 38 },
            new float[] { 43, 49 },
            new float[] { 52, 43 },
            new float[] { 70, 64 },
            new float[] { 38, 32 },
            new float[] { 28, 43 },
            new float[] { 44, 46 },
            new float[] { 40, 53 },
            new float[] { 28, 64 },
            new float[] { 40, 49 },
            new float[] { 55, 44 },
            new float[] { 44, 47 },
            new float[] { 28, 47 },
            new float[] { 37, 57 },
            new float[] { 33, 49 },
            new float[] { 74, 63 },
            new float[] { 44, 46 },
            new float[] { 44, 43 },
            new float[] { 40, 39 },
            new float[] { 44, 53 },
            new float[] { 52, 53 },
            new float[] { 70, 63 },
            new float[] { 40, 60 },
            new float[] { 36, 46 },
            new float[] { 33, 43 },
            new float[] { 33, 43 },
            new float[] { 33, 43 },
            new float[] { 55, 43 },
            new float[] { 63, 46 },
            new float[] { 66, 56 },
            new float[] { 52, 67 },
            new float[] { 55, 54 },
            new float[] { 36, 35 },
            new float[] { 48, 32 },
            new float[] { 55, 39 },
            new float[] { 59, 60 },
            new float[] { 55, 63 },
            new float[] { 66, 70 },
            new float[] { 55, 54 },
            new float[] { 48, 60 },
            new float[] { 55, 60 },
            new float[] { 52, 56 },
            new float[] { 48, 39 },
            new float[] { 52, 63 },
            new float[] { 52, 56 },
            new float[] { 44, 43 },
            new float[] { 63, 56 },
            new float[] { 33, 35 },
            new float[] { 59, 56 },
            new float[] { 52, 54 },
            new float[] { 44, 56 },
            new float[] { 52, 56 },
            new float[] { 55, 54 },
            new float[] { 52, 43 },
            new float[] { 63, 63 },
            new float[] { 48, 49 },
            new float[] { 40, 35 },
            new float[] { 48, 43 },
            new float[] { 55, 54 },
            new float[] { 48, 56 },
            new float[] { 59, 60 },
            new float[] { 52, 49 },
            new float[] { 44, 56 },
            new float[] { 63, 46 }
        };

        public static float[][] testDataNet = new float[][]
        {
            new float[] { 62,  56,  58,  50,  52 },
            new float[] { 69,  56,  67,  55,  68 },
            new float[] { 35,  35,  49,  45,  44 },
            new float[] { 43,  46,  35,  38,  39 },
            new float[] { 62,  46,  58,  72,  60 },
            new float[] { 57,  46,  73,  66,  68 },
            new float[] { 50,  49,  49,  45,  44 },
            new float[] { 57,  42,  40,  45,  48 },
            new float[] { 57,  53,  63,  72,  57 },
            new float[] { 73,  66,  49,  66,  65 }
        };


        public static float[][] testAnswersNet = new float[][]
        {
            new float[] { 59, 54 },
            new float[] { 63, 54 },
            new float[] { 36, 46 },
            new float[] { 44, 46 },
            new float[] { 52, 60 },
            new float[] { 55, 70 },
            new float[] { 52, 49 },
            new float[] { 40, 54 },
            new float[] { 59, 56 },
            new float[] { 55, 67 }
        };
    }
}
