using System;
using System.Collections.Generic;
using System.Linq;
using Roslyn.Compilers;
using Roslyn.Compilers.Common;
using Roslyn.Compilers.CSharp;

namespace CudaSharpCompiler
{
    static class RoslynInteraction
    {
        public static IEnumerable<Tuple<MethodDeclarationSyntax, ISemanticModel>> GetKernels(IEnumerable<string> files)
        {
            var compilation = Compilation.Create("Kernel.dll", syntaxTrees: files.Select(f => SyntaxTree.ParseFile(f)));
            var kernels = compilation.SyntaxTrees.SelectAsArray(compilation.GetSemanticModel).SelectMany(FindKernels);
            return kernels;
        }

        static IEnumerable<Tuple<MethodDeclarationSyntax, ISemanticModel>> FindKernels(ISemanticModel semantic)
        {
            var methods = semantic.SyntaxTree.GetRoot().DescendantNodes().OfType<MethodDeclarationSyntax>();
            var kernels = methods.Where(syntax => ((IMethodSymbol)semantic.GetDeclaredSymbol(syntax)).IsKernel());
            var tuples = kernels.Select(m => Tuple.Create(m, semantic));
            return tuples;
        }
    }
}