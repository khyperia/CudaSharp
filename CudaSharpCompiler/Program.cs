using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Roslyn.Compilers.Common;

namespace CudaSharpCompiler
{
    static class CudaSharp
    {

        static void Main(string[] args)
        {
            File.WriteAllBytes("kernel.ptx", Compile(args));
        }

        static byte[] Compile(IEnumerable<string> files)
        {
            var kernels = RoslynInteraction.GetKernels(files);
            var module = LlvmInteraction.EmitModule(kernels);
            return module;
        }

        public static bool IsKernel(this IMethodSymbol method)
        {
            return method.GetAttributes().Any(cad =>
            {
                var name = cad.AttributeClass.Name.ToLower();
                var substr = name.LastIndexOf('.');
                if (substr >= 0)
                    name = name.Substring(substr + 1);
                return name == "kernel" || name == "kernelattribute";
            });
        }
    }
}
