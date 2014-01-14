using System;
using ManagedCuda;

namespace CudaSharpTest
{
    class Program
    {
        static void Main()
        {
            Console.WriteLine("Unsupported instructions:");
            foreach (var opCode in CudaSharp.CudaSharp.UnsupportedInstructions)
                Console.WriteLine(opCode);
            var ptx = CudaSharp.CudaSharp.Translate<int[]>(kernel);
            Test(ptx);
            Console.ReadKey(true);
        }

        // ReSharper disable once InconsistentNaming
        static void kernel(int[] arr)
        {
            var val = arr[0];
            if (val != 0)
                arr[0] = val + 3;
        }

        static void Test(string ptxFile)
        {
            var context = new CudaContext();
            var kernel = context.LoadKernelPTX(ptxFile, "kernel");
            var memory = context.AllocateMemory(4);
            var gpuMemory = new CudaDeviceVariable<int>(memory);
            var cpuMemory = new int[1];
            cpuMemory[0] = 2;
            gpuMemory.CopyToDevice(cpuMemory);
            kernel.BlockDimensions = 1;
            kernel.GridDimensions = 1;
            kernel.Run(memory);
            gpuMemory.CopyToHost(cpuMemory);
            Console.WriteLine(cpuMemory[0]);
        }
    }
}
