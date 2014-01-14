using System;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Emit;
using LLVM;
using LLVM.NativeLibrary;

namespace CudaSharp
{
    public class CudaSharp
    {
        static CudaSharp()
        {
            LLVMDLL.Load();
        }

        public static OpCode[] UnsupportedInstructions
        {
            get { return Translator.UnsupportedInstructions; }
        }

        public static string Translate(Action method) { return Translate(method.Method); }
        public static string Translate<T1>(Action<T1> method) { return Translate(method.Method); }
        public static string Translate<T1, T2>(Action<T1, T2> method) { return Translate(method.Method); }
        public static string Translate<T1, T2, T3>(Action<T1, T2, T3> method) { return Translate(method.Method); }
        public static string Translate<T1, T2, T3, T4>(Action<T1, T2, T3, T4> method) { return Translate(method.Method); }
        public static string Translate<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> method) { return Translate(method.Method); }
        public static string Translate<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> method) { return Translate(method.Method); }

        public static string Translate(MethodInfo method)
        {
            Target.InitializeNative();

            var module = Translator.Translate(Context.Global, method);

            if (PInvoke.LLVMWriteBitcodeToFile(module, "fzoo.bc") != 0)
                throw new Exception("LLVMWriteBitcodeToFile returned nonzero");

            const string outfile = "fzoo.ptx";
            InvokeLlc("fzoo.bc", outfile);
            return outfile;
        }

        private static void InvokeLlc(string inputFile, string outputFile)
        {
            var llc = Process.Start(new ProcessStartInfo("llc", string.Format("-mcpu=sm_20 {0} -o {1}", inputFile, outputFile)) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true });
            var stdout = llc.StandardOutput.ReadToEnd();
            var stderr = llc.StandardError.ReadToEnd();
            if (string.IsNullOrWhiteSpace(stdout) == false)
                throw new Exception(stdout);
            if (string.IsNullOrWhiteSpace(stderr) == false)
                throw new Exception(stderr);
            llc.WaitForExit();
        }
    }
}