using System;
using System.Text;
using CudaSharp;
using ManagedCuda;

namespace CudaSharpTest
{
    static class Program
    {
        static void Main()
        {
            CudaSharp.CudaSharp.Translate();
            var ptx = CudaSharp.CudaSharp.Translate<int[]>(kernel);
            PrintPtx(ptx);
            Test(ptx);
            Console.ReadKey(true);
        }

        struct testStruct
        {
            public testStruct(int fzoo)
            {
                x = fzoo;
            }

            public int x;
        }

        static void store(int[] arr, int value)
        {
            arr[Gpu.ThreadX() + Gpu.BlockX() * Gpu.ThreadDimX()] = value;
        }

        static void kernel(int[] arr)
        {
            var tid = Gpu.ThreadX() + Gpu.BlockX() * Gpu.ThreadDimX();
            var val = arr[tid];

            if (val != 0)
            {
                var strct = new testStruct();
                strct.x += 2;
                val = strct.x;
                store(arr, val + 3);
            }
        }

        static void Test(byte[] ptxFile)
        {
            const int size = 16;
            var context = new CudaContext();
            var kernel = context.LoadKernelPTX(ptxFile, "kernel");
            var memory = context.AllocateMemory(4 * size);
            var gpuMemory = new CudaDeviceVariable<int>(memory);
            var cpuMemory = new int[size];
            for (var i = 0; i < size; i++)
                cpuMemory[i] = i - 2;
            gpuMemory.CopyToDevice(cpuMemory);
            kernel.BlockDimensions = 4;
            kernel.GridDimensions = 4;
            kernel.Run(memory);
            gpuMemory.CopyToHost(cpuMemory);
            for (var i = 0; i < size; i++)
                Console.WriteLine("{0} = {1}", i, cpuMemory[i]);
        }

        static void PrintPtx(byte[] bytes)
        {
            Console.Write(Encoding.UTF8.GetString(bytes));
        }
    }
}
