using System;
using ManagedCuda;
using NUnit.Framework;

namespace CudaSharpTest
{
    class UnitTests
    {
        private CudaContext _context;
        
        [SetUp]
        public void Initialize()
        {
            _context = new CudaContext();
        }

        private T[] RunKernel<T>(Action<T[]> method, T[] parameters) where T : struct
        {
            var methodInfo = method.Method;
            string[] kernels;
            var ptx = CudaSharp.CudaSharp.Translate(out kernels, methodInfo);
            Console.WriteLine(kernels[0]);
            var kernel = _context.LoadKernelPTX(ptx, kernels[0]);
            var maxThreads = kernel.MaxThreadsPerBlock;
            if (parameters.Length <= maxThreads)
            {
                kernel.BlockDimensions = parameters.Length;
                kernel.GridDimensions = 1;
            }
            else
            {
                kernel.BlockDimensions = maxThreads;
                kernel.GridDimensions = parameters.Length / maxThreads;
                if ((kernel.BlockDimensions * kernel.GridDimensions) != parameters.Length)
                    throw new Exception(string.Format("Invalid parameters size (must be <= {0} or a multiple of {0}", maxThreads));
            }
            var gpuMem = new CudaDeviceVariable<T>(parameters.Length);
            gpuMem.CopyToDevice(parameters);
            kernel.Run(gpuMem.DevicePointer);
            gpuMem.CopyToHost(parameters);
            gpuMem.Dispose();
            return parameters;
        }

        private T RunKernel<T>(Action<T[]> method, T parameter) where T : struct
        {
            return RunKernel(method, new[] { parameter })[0];
        }

        [Test]
        public void Add()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] += 2, 2));
        }

        private static int CallTest(int i)
        {
            return i + 2;
        }

        [Test]
        public void Call()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = CallTest(p[0]), 2));
        }
    }
}