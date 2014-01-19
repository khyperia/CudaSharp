using System;
using System.Linq;
using CudaSharp;
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
            string llvmIr, ptxIr;
            var ptx = CudaSharp.CudaSharp.Translate(out kernels, out llvmIr, out ptxIr, "sm_20", methodInfo);
            Console.WriteLine(llvmIr);
            Console.WriteLine(ptxIr);
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

        struct SingleValueStruct
        {
            public SingleValueStruct(int x)
            {
                X = x;
            }

            public int X;

            public int GetX()
            {
                return X;
            }
        }

        private static SingleValueStruct StructByValTest(SingleValueStruct obj)
        {
            obj.X = obj.X + 2;
            return obj;
        }

        [Test, Ignore]
        public void StructByValue()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = StructByValTest(new SingleValueStruct(p[0])).X, 2));
        }

        private static SingleValueStruct StructReturnTest(int n)
        {
            return new SingleValueStruct(n + 2);
        }

        [Test]
        public void StructReturn()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = StructReturnTest(p[0]).X, 2));
        }

        [Test]
        public void StructInstanceMethod()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = StructReturnTest(p[0]).GetX(), 2));
        }

        [Test]
        public void StructInstanceMethodInline()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = new SingleValueStruct(p[0] + 2).GetX(), 2));
        }

        private static void StructByRefTest(ref SingleValueStruct obj)
        {
            obj.X = obj.X + 2;
        }

        [Test]
        public void StructByRef()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                var obj = new SingleValueStruct(p[0]);
                StructByRefTest(ref obj);
                p[0] = obj.X;
            }, 2));
        }

        [Test]
        public void IfStatement()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                if (p[0] == 2)
                    p[0] = 4;
            }, 2));
        }

        [Test]
        public void IfNotStatement()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                if (p[0] != 1)
                    p[0] = 4;
            }, 2));
        }

        [Test]
        public void IfCltStatement()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                if (p[0] < 4)
                    p[0] = 4;
            }, 2));
        }

        [Test]
        public void WhileStatement()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                while (p[0] != 4)
                    p[0]++;
            }, 2));
        }

        [Test]
        public void ForStatement()
        {
            Assert.AreEqual(4, RunKernel(p =>
            {
                for (var i = 0; i < 2; i++)
                    p[0]++;
            }, 2));
        }

        [Test]
        public void ThreadIdxIntrinsics()
        {
            Assert.AreEqual(Enumerable.Range(0, 256).ToArray(), RunKernel(p =>
            {
                var tid = Gpu.BlockX() * Gpu.ThreadDimX() + Gpu.ThreadX();
                p[tid] = tid;
            }, new int[256]));
        }

        private static int IntCastingTest(short x)
        {
            return x;
        }

        [Test, Ignore]
        public void IntCasting()
        {
            Assert.AreEqual(4, RunKernel(p => p[0] = IntCastingTest((short)p[0]) + 2, 2));
        }
    }
}
