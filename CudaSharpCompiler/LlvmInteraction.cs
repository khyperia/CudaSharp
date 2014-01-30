using System;
using System.Collections.Generic;
using System.IO;
using CudaSharp;
using LLVM;
using Roslyn.Compilers;
using Roslyn.Compilers.Common;
using Roslyn.Compilers.CSharp;
using Type = LLVM.Type;

namespace CudaSharpCompiler
{
    static class LlvmInteraction
    {
        static LlvmInteraction()
        {
            var extractTo = Path.GetFullPath("LLVM-3.3");
            if (File.Exists(extractTo))
                return;
            var file = File.Open(extractTo, FileMode.OpenOrCreate);
            var llvm34 = System.Reflection.Assembly.GetExecutingAssembly().GetManifestResourceStream("CudaSharp.LLVM-3.4.dll");
            if (llvm34 == null)
                throw new Exception("Could not extract LLVM-3.4.dll");
            llvm34.CopyTo(file);
            file.Close();
        }

        public static byte[] EmitModule(IEnumerable<Tuple<MethodDeclarationSyntax, ISemanticModel>> kernels)
        {
            var module = new Module("Module", Context.Global);

            if (Environment.Is64BitOperatingSystem)
            {
                module.SetTarget("nvptx64-nvidia-cuda");
                module.SetDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            }
            else
            {
                module.SetTarget("nvptx-nvidia-cuda");
                module.SetDataLayout("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            }

            foreach (var kernel in kernels)
                Translate(module, kernel.Item1, kernel.Item2);

            var ptx = PInvokeHelper.EmitInMemory(module, "sm_35");
            //llvmIr = Marshal.PtrToStringAnsi(PInvoke.LLVMPrintModuleToString(module));
            //ptxIr = Encoding.UTF8.GetString(ptx);
            return ptx;
        }

        private static void Translate(Module module, MethodDeclarationSyntax methodSyntax, ISemanticModel semanticModel)
        {
            var methodSymbol = (IMethodSymbol)semanticModel.GetDeclaredSymbol(methodSyntax);
            var fntype = new FunctionType(module.TranslateType(methodSymbol.ReturnType),
                methodSymbol.Parameters.SelectAsArray(p => module.TranslateType(p.Type)).ToArray());
            var function = module.CreateFunction(methodSymbol.Name, fntype);
            var body = methodSyntax.Body;
            body.Accept(new LlvmSyntaxVisitor(module, function, methodSymbol));
        }

        private static Type TranslateType(this Module module, ITypeSymbol returnType)
        {
            
        }
    }

    class LlvmSyntaxVisitor : SyntaxVisitor<Value>
    {
        private readonly Module _module;
        private readonly Function _function;
        private InstructionBuilder _currentBlock;
        private Value[] _parameters;

        public LlvmSyntaxVisitor(Module module, Function function, IMethodSymbol methodSymbol)
        {
            _module = module;
            _function = function;
            _currentBlock = new InstructionBuilder(module.Context, new Block("", module.Context, function));
            _parameters = new Value[methodSymbol.Parameters.Count];
            for (var i = 0; i < _parameters.Length; i++)
            {
                var value = function[i];
                var alloca = _currentBlock.StackAlloc(value.Type);
                _currentBlock.Store(value, alloca);
                _parameters[i] = alloca;
            }
        }
    }
}