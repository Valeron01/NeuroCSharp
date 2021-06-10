using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace NeuralNetwork
{
    class Program
    {
        static void Zero(double[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.234;
            }
        }
        static double[] PixelsArray(string path)
        {
            List<double> arr = new List<double>(784);

            Image image = Image.FromFile(path);

            Bitmap bitmap = new Bitmap(image);

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    arr.Add(getBW(bitmap.GetPixel(x, y)));
                }
            }

            double getBW(Color color)
            {
                int d = color.R;
                d += color.B;
                d += color.G;
                return d / 3.0;
            }

            for (int i = 0; i < 784; i++)
            {
                if (arr[i] >= 128)
                    arr[i] = 1;
                else
                    arr[i] = 0;
            }

            return arr.ToArray();
        }
        static void Main(string[] args)
        {
            const int numOfNumbers = 3200;

            NeuralNetwork network = new NeuralNetwork(784, 16, 16, 10);

            double[][] data = new double[4][]
            {
                new double[2]{0, 0},
                new double[2]{1, 0},
                new double[2]{1, 1},
                new double[2]{0, 1},
            };

            double[] answers = new double[4]
            {
                0,1,0,1
            };




            #region Подготовка тренировочных данных(Все массивы)
            double[][] Zero = new double[numOfNumbers][];
            double[][] One = new double[numOfNumbers][];
            double[][] Two = new double[numOfNumbers][];
            double[][] Three = new double[numOfNumbers][];
            double[][] Four = new double[numOfNumbers][];
            double[][] Five = new double[numOfNumbers][];
            double[][] Six = new double[numOfNumbers][];
            double[][] Seven = new double[numOfNumbers][];
            double[][] Eight = new double[numOfNumbers][];
            double[][] Nine = new double[numOfNumbers][];

            double[] answerZero = new double[10]
            {
                1,0,0,0,0,0,0,0,0,0
            };
            double[] answerOne = new double[10]
            {
                0,1,0,0,0,0,0,0,0,0
            };
            double[] answerTwo = new double[10]
            {
                0,0,1,0,0,0,0,0,0,0
            };
            double[] answerThree = new double[10]
            {
                0,0,0,1,0,0,0,0,0,0
            };
            double[] answerFour = new double[10]
            {
                0,0,0,0,1,0,0,0,0,0
            };
            double[] answerFive = new double[10]
            {
                0,0,0,0,0,1,0,0,0,0
            };
            double[] answerSix = new double[10]
            {
                0,0,0,0,0,0,1,0,0,0
            };
            double[] answerSeven = new double[10]
            {
                0,0,0,0,0,0,0,1,0,0
            };
            double[] answerEight = new double[10]
            {
                0,0,0,0,0,0,0,0,1,0
            };
            double[] answerNine = new double[10]
            {
                0,0,0,0,0,0,0,0,0,1
            };

            double[][] numberAnswers = new double[10][]
            {
                answerZero,
                answerOne,
                answerTwo,
                answerThree,
                answerFour,
                answerFive,
                answerSix,
                answerSeven,
                answerEight,
                answerNine
            };
            #endregion

            ///////////////////////////Нужно подобрать методику обучения


            network.Load("network_numbers_eueurgs");
            

            


            DigitRecognizer recognizer = new DigitRecognizer(network);

            for (int i = 0; i < 10; i++)
            {
                double[] arr = PixelsArray($"training/{i}/2.png");
                Console.WriteLine($"Нейросеть думает, что это цифра {recognizer.GetDigit(arr)}, но это цифра {i}");
            }
            double[] result = network.GetOutputArray(PixelsArray($"training/9/107.png"));


            foreach (var item in result)
            {
                Console.WriteLine(item);
            }
            Console.ReadKey();

            return;

            Random r = new Random();

            Console.WriteLine("Начинаем собирать данные!");

            for (int i = 2000; i < numOfNumbers; i++)
            {
                Zero[i] = PixelsArray($"training/0/{i}.png");
                One[i] = PixelsArray($"training/1/{i}.png");
                Two[i] = PixelsArray($"training/2/{i}.png");
                Three[i] = PixelsArray($"training/3/{i}.png");
                Four[i] = PixelsArray($"training/4/{i}.png");
                Five[i] = PixelsArray($"training/5/{i}.png");
                Six[i] = PixelsArray($"training/6/{i}.png");
                Seven[i] = PixelsArray($"training/7/{i}.png");
                Eight[i] = PixelsArray($"training/8/{i}.png");
                Nine[i] = PixelsArray($"training/9/{i}.png");
            }//Заполняем массивы

            double[][][] allNumbers = new double[10][][]
            {
                Zero,
                One,
                Two,
                Three,
                Four,
                Five,
                Six,
                Seven,
                Eight,
                Nine
            };


            Console.WriteLine("Данные получены!");

            for (int i = 0; i < 784; i++)
            {
                if (i % 28 == 0)
                    Console.Write("\n");
                Console.Write(Six[2004][i]);
            }
            Console.ReadKey();




            for (int i = 2000; i < numOfNumbers; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    network.Train(allNumbers[j][i], numberAnswers[j]);
                }

                Console.WriteLine($"I: {i}");
            }


            for (int j = 0; j < 10; j++)
            {
                Console.WriteLine($"-------- {j} ---------");
                double[] actually = network.GetOutputArray(allNumbers[j][2858]);
                for (int l = 0; l < 10; l++)
                {
                    Console.WriteLine($"Должно быть: {numberAnswers[j][l]}, однако {actually[l]}");
                }
            }

            network.Save("network_numbers_eueurgs");

            //network.Load("network_numbers_1000_iterations");



            Console.ReadKey();
        }
    }

    class Neuron : ICloneable
    {
        static Random r = new Random();

        readonly NeuronType neuronType;

        public double learningRate = 0.7;

        public double Bias = 0;

        public double[] Weights;

        double[] inputs;

        public double[] Inputs
        {
            get { return inputs; }
            set
            {
                if (value.Length != Dendrits)
                    throw new Exceptions.InvalidSizeOfArray();
                inputs = value;
            }
        }

        public int Dendrits;

        public double DC_DA_HIDDEN = 0;

        public double Output
        {
            get
            {
                return Sigmoid(Z);
            }
        }

        public double Z
        {
            get
            {
                double z = Bias;
                for (int i = 0; i < Weights.Length; i++)
                {
                    z += Weights[i] * inputs[i];
                }

                return z;
            }
        }

        public double Error { get; private set; }

        public Neuron(int dendrits, NeuronType neuronType)
        {
            this.neuronType = neuronType;
            Dendrits = dendrits;

            Weights = new double[dendrits];
            Inputs = new double[dendrits];

            for (int i = 0; i < Dendrits; i++)
            {
                Weights[i] = r.NextDouble() * ((r.NextDouble() > 0.5) ? -1 : 1);
            }
            Bias = r.NextDouble();
        }

        /// <summary>
        /// Тренировка нейрона выходного слоя
        /// </summary>
        /// <param name="desired"></param>
        public void Train(double desired)
        {
            Error = 2 * (Output - desired);

            for (int i = 0; i < Inputs.Length; i++)
            {
                double dW = Inputs[i] * SigmoidDX() * Error;
                Weights[i] -= learningRate * dW;
            }
            Bias -= SigmoidDX() * Error * learningRate;
        }

        /// <summary>
        /// Тренировка нейрона скрытого слоя
        /// </summary>
        /// <param name="nextLayer">Массив, содержащий нейроны следующего слоя</param>
        /// <param name="myIndex">Индекс нейрона в его слое</param>
        public void Train(Neuron[] nextLayer, int myIndex)
        {
            Error = 0;
            foreach (Neuron neuron in nextLayer)
            {
                Error += neuron.Weights[myIndex] * neuron.SigmoidDX() * neuron.Error;
            }

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] -= inputs[i] * SigmoidDX() * Error * learningRate;
            }
            Bias -= SigmoidDX() * Error;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////

        double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        public double SigmoidDX()
        {
            double x = Z;
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        public object Clone()
        {
            Neuron ret = new Neuron(Dendrits, neuronType)
            {
                Weights = Weights,
                Bias = Bias,
                inputs = inputs
            };
            return ret;
        }
    }

    enum NeuronType
    {
        Input,
        Hidden,
        Output
    }

    class NeuralNetwork
    {
        public readonly int NumberOfLayers = 0;
        public Neuron[][] Neurons;
        public readonly int[] LayerInfo;

        readonly bool multiplyOutputLayer;

        /// <summary>
        /// Создает экземпляр нейросети
        /// </summary>
        /// <param name="LayerInfo">Количество нейронов в слое, длина массива - количесество слоев.(вместе с входным слоем)</param>
        public NeuralNetwork(params int[] LayerInfo)
        {
            this.LayerInfo = LayerInfo;
            NumberOfLayers = LayerInfo.Length - 1;
            Neurons = new Neuron[NumberOfLayers][];
            for (int i = 0; i < NumberOfLayers; i++)
            {
                Neurons[i] = new Neuron[LayerInfo[i + 1]];
            }

            for (int i = 0; i < NumberOfLayers; i++)
            {
                for (int j = 0; j < LayerInfo[i + 1]; j++)
                {
                    if (i == NumberOfLayers - 1)
                    {
                        Neurons[i][j] = new Neuron(LayerInfo[i], NeuronType.Output);
                        Console.WriteLine("Instantiated output neuron!!!");
                    }
                    else
                    {
                        Neurons[i][j] = new Neuron(LayerInfo[i], NeuronType.Hidden);
                        Console.WriteLine("Instantiate Hidden Neuron!");
                    }
                }
            }
            if (Neurons[Neurons.Length - 1].Length != 1)
                multiplyOutputLayer = true;
        }


        /// <summary>
        /// Выдает Выходное знаяение нейросети
        /// </summary>
        /// <param name="inputs">Вход для нейросети</param>
        /// <returns></returns>
        public double GetOutput(double[] inputs)
        {
            if (multiplyOutputLayer)
                throw new Exception("Выбранная перегрузка метода Train() не подходит для данной нейросети");
            SetInput(inputs);
            return Neurons[Neurons.Length - 1][0].Output;
        }
        public double[] GetOutputArray(double[] inputs)
        {
            if (!multiplyOutputLayer)
                throw new Exception("Выбранная перегрузка метода Train() не подходит для данной нейросети");
            SetInput(inputs);

            double[] ret = new double[Neurons[Neurons.Length - 1].Length];
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = Neurons[Neurons.Length - 1][i].Output;
            }

            return ret;
        }

        private void SetInput(double[] inputs)
        {
            if (inputs.Length != LayerInfo[0])
            {
                LayerInfo[0] = inputs.Length;
                throw new Exceptions.BadAlgorithm();
            }

            foreach (Neuron singleNeuron in Neurons[0])
            {
                singleNeuron.Inputs = inputs;
            }

            for (int i = 1; i < NumberOfLayers; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    Neuron n = Neurons[i][j];
                    Neuron[] arr = Neurons[i - 1];
                    for (int k = 0; k < arr.Length; k++)
                    {
                        n.Inputs[k] = arr[k].Output;
                    }
                }
            }
        }

        public void Train(double[] input, double desired)
        {
            if (multiplyOutputLayer)
                throw new Exception("Выбранная перегрузка метода Train() не подходит для данной нейросети");
            SetInput(input);

            Neurons[Neurons.Length - 1][0].Train(desired);

            for (int layerIndex = Neurons.Length - 2; layerIndex >= 0; layerIndex--)
            {
                Neuron[] nextLayer = Neurons[layerIndex + 1];
                Neuron[] curLayer = Neurons[layerIndex];

                for (int i = 0; i < curLayer.Length; i++)
                {
                    curLayer[i].Train(nextLayer, i);
                }
            }
        }

        public void Train(double[] input, double[] desired)
        {
            if (!multiplyOutputLayer)
                throw new Exception("Выбранная перегрузка метода Train() не подходит для данной нейросети");
            SetInput(input);

            Neuron[] outputLayer = Neurons[Neurons.Length - 1];

            if (desired.Length != outputLayer.Length)
                throw new Exception("Массив решений не для подходит для текущей нейросети");

            for (int i = 0; i < outputLayer.Length; i++)
            {
                outputLayer[i].Train(desired[i]);
            }

            for (int layerIndex = Neurons.Length - 2; layerIndex >= 0; layerIndex--)
            {
                Neuron[] nextLayer = Neurons[layerIndex + 1];
                Neuron[] curLayer = Neurons[layerIndex];

                for (int i = 0; i < curLayer.Length; i++)
                {
                    curLayer[i].Train(nextLayer, i);
                }
            }
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(path);
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    for (int k = 0; k < Neurons[i][j].Weights.Length; k++)
                    {
                        double weight = Neurons[i][j].Weights[k];
                        File.AppendAllText($"{path}/weights.bin", $"{weight}\n");
                    }
                    double bias = Neurons[i][j].Bias;
                    File.AppendAllText($"{path}/biases.bin", $"{bias}\n");
                }
            }
        }
        public void Load(string path)
        {
            long lineWeights = 0;
            long lineBiases = 0;

            string[] weights = File.ReadAllLines($"{path}/weights.bin");
            string[] biases = File.ReadAllLines($"{path}/biases.bin");

            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    for (int k = 0; k < Neurons[i][j].Weights.Length; k++)
                    {
                        double weight = Double.Parse(weights[lineWeights]);
                        Neurons[i][j].Weights[k] = weight;
                        lineWeights++;
                    }
                    double bias = Double.Parse(biases[lineBiases]);
                }
            }
        }
    }

    namespace Exceptions
    {
        class BadAlgorithm : Exception
        {
            public override string Message => "Размер назначаемого массива не соответствует шаблону!!!";
        }
        class InvalidSizeOfArray : Exception
        {
            public override string Message => "Размер массива, присваемого ко входу нейрона не соответствует шаблону!";
        }
    }

    class NeuralNetworkWorker
    {
        public readonly NeuralNetwork neuralNetwork;
        public NeuralNetworkWorker(NeuralNetwork network)
        {
            neuralNetwork = network;
        }
    }

    class DigitRecognizer
    {
        readonly NeuralNetwork neuralNetwork;
        public DigitRecognizer(NeuralNetwork neuralNetwork)
        {
            this.neuralNetwork = neuralNetwork;
        }

        public int GetDigit(double[] inputs)
        {
            int ret = 0;
            double max = 0;

            double[] answer = neuralNetwork.GetOutputArray(inputs);

            for (int i = 0; i < 10; i++)
            {
                if (max < answer[i])
                {
                    max = answer[i];
                    ret = i;
                }
            }

            return ret;
        }

    }

}

/*



    Neuron n = new Neuron(1, NeuronType.Output);

            for (int i = 0; i < 1000; i++)
            {
                if (i % 2 == 0)
                {
                    n.Inputs[0] = 0;
                    n.Train(1);
                }
                else
                {
                    n.Inputs[0] = 1;
                    n.Train(0);
                }
                Console.WriteLine(i);
            }

            n.Inputs[0] = 1;
            Console.WriteLine("NOT 1 is " + n.Output);

            n.Inputs[0] = 0;
            Console.WriteLine("NOT 0 is " + n.Output);

            n.Inputs[0] = 1;

            Console.WriteLine(n.Output);

            Console.ReadKey();
            return;


     Neuron not = new Neuron(1);
            for (int i = 0; i < 1000; i++)
            {
                if (i % 2 == 0)
                {
                    not.Inputs[0] = 0;
                    not.Train(1);
                    Console.WriteLine("NOT 0 is " + not.Output);
                }
                else
                {
                    not.Inputs[0] = 1;
                    not.Train(0);
                    Console.WriteLine("NOT 1 is " + not.Output);
                }
            }
            Console.ReadKey();
            return;



    public void Train(double desired, Neuron[] previousLayer)
        {
            if (neuronType == NeuronType.Output)
            {
                double output = Output;

                double dC0_dA = 2 * (output - desired);
                double dA_dZ = SigmoidDX(Z);
                double dZ_dW = 0;
                double dC0_dW = 0;
                double dC0_dB = dC0_dA * dA_dZ;

                for (int i = 0; i < Weights.Length; i++)
                {
                    dZ_dW = inputs[i];
                    dC0_dW = dA_dZ * dZ_dW * dC0_dA;
                    Weights[i] -= dC0_dW * learningRate;
                }
                Bias -= dC0_dB * learningRate;

                ////Позаботимся о предыдущих слоях)))
                for (int i = 0; i < previousLayer.Length; i++)
                {
                    Neuron previousNeuron = previousLayer[i];
                    previousNeuron.DC_DA_HIDDEN += Weights[i] * dA_dZ * dC0_dA;
                }
                return;
            }
            else if(neuronType == NeuronType.Hidden)
            {
                double dC0_dW = 0;
                for (int i = 0; i < Weights.Length; i++)
                {
                    dC0_dW = inputs[i] * SigmoidDX(Z) * DC_DA_HIDDEN;
                    Weights[i] -= dC0_dW * learningRate;
                   // Console.WriteLine("TRAIN");
                }
                Bias -= SigmoidDX(Z) * DC_DA_HIDDEN;
            }
        }

        public void Train()
        {
            if (neuronType == NeuronType.Hidden)
            {
                double dC0_dW = 0;
                for (int i = 0; i < Weights.Length; i++)
                {
                    dC0_dW = inputs[i] * SigmoidDX(Z) * DC_DA_HIDDEN;
                    Weights[i] -= dC0_dW * learningRate;
                    //Console.WriteLine("TRAIN");
                }
                Bias -= SigmoidDX(Z) * DC_DA_HIDDEN;
                DC_DA_HIDDEN = 0;
            }
        }



    double[] Zero = PixelsArray("Zero.bmp");
            double[] One = PixelsArray("One.bmp");
            double[] TestZero = PixelsArray("Zero.bmp");

            double[][] Images = new double[2][] { Zero, One };
            double[] im = new double[2] { 0, 1 };

            Neuron ImageRaspozznavatel = new Neuron(784, NeuronType.Output);
            

            for (int epoch = 0; epoch < 1000; epoch++)
            {
                for (int i = 0; i < 2; i++)
                {
                    ImageRaspozznavatel.Inputs = Images[i];
                    ImageRaspozznavatel.Train(im[i]);
                    Console.WriteLine($"Should be: {im[i]}, actually: {ImageRaspozznavatel.Output}");
                }
            }
            Console.WriteLine("////////////////////");

            ImageRaspozznavatel.Inputs = One;

            Console.WriteLine("Should be 0, acyually: " + ImageRaspozznavatel.Output);


            Console.ReadKey();
            return;








     */


/* return;
         double[][] trainData = new double[6][]
         {
             new double[3]{1, 1, 1},
             new double[3]{1, 1, 0},
             new double[3]{1, 0, 0},
             new double[3]{0, 0, 0},
             new double[3]{0, 0, 1},
             new double[3]{0, 1, 1}
         };

         double[] trainDataAnswers = new double[6] { 1, 1, 0, 0, 1, 1 };

         NeuralNetwork neuralNetwork = new NeuralNetwork(2, 2, 1);

         //Console.ReadKey();
         for (int epoch = 0; epoch < 3000; epoch++)
         {
             for (int i = 0; i < 4; i++)
             {
                 neuralNetwork.Train(data[i], answers[i]);
                 //Console.WriteLine($"{data[i][0]} XOR {data[i][1]} is {neuralNetwork.GetOutput(data[i])}");
             }
             //Console.WriteLine("Bias: " + neuralNetwork.Neurons[1][0].Bias);
         }

         for (int i = 0; i < 4; i++)
         {
             neuralNetwork.Train(data[i], answers[i]);
             Console.WriteLine($"{data[i][0]} XOR {data[i][1]} is {neuralNetwork.GetOutput(data[i])}");
         }




    n.Inputs[0] = 1;
            Console.WriteLine("NOT 1 is " + n.Output);

            n.Inputs[0] = 0;
            Console.WriteLine("NOT 0 is " + n.Output);

            n.Inputs[0] = 0;

            Console.WriteLine(n.Output);

            Console.ReadKey();


            Neuron n1 = new Neuron(2, NeuronType.Hidden);
            Neuron n2 = new Neuron(2, NeuronType.Hidden);
            Neuron o = new Neuron(2, NeuronType.Output);

            for (int epoch = 0; epoch < 2000; epoch++)
            {
                for (int i = 0; i < 4; i++)
                {
                    n1.Inputs = data[i];
                    n2.Inputs = data[i];
                    o.Inputs = new double[2] { n1.Output, n2.Output };
                    o.Train(answers[i]);

                    double dC_dA = o.Weights[0] * o.SigmoidDX(o.Z) * 2 * (o.Output - answers[i]);

                    for (int j = 0; j < 2; j++)
                    {
                        n1.Weights[j] -= dC_dA * n1.learningRate;
                    }
                    n1.Bias -= n1.SigmoidDX(n1.Z) * dC_dA * n2.learningRate;

                    dC_dA = o.Weights[1] * o.SigmoidDX(o.Z) * 2 * (o.Output - answers[i]);

                    for (int j = 0; j < 2; j++)
                    {
                        n2.Weights[j] -= dC_dA * n2.learningRate;
                    }

                    n2.Bias -= n2.SigmoidDX(n2.Z) * dC_dA * n2.learningRate;

                    //Console.WriteLine($"{data[i][0]} XOR {data[i][1]} is {o.Output}");
                }
            }

            for (int i = 0; i < 4; i++)
            {
                n1.Inputs = data[i];
                n2.Inputs = data[i];
                o.Inputs = new double[2] { n1.Output, n2.Output };

                Console.WriteLine($"{data[i][0]} XOR {data[i][1]} is {o.Output}");
            }
            Console.ReadKey();









         Console.ReadKey();*/
