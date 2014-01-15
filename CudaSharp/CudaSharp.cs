using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Reflection.Emit;
using LLVM;
using LLVM.NativeLibrary;

namespace CudaSharp
{
    public static class CudaSharp
    {
        static CudaSharp()
        {
            //LLVMDLL.Load();
            var extractTo = Path.GetFullPath("LLVM-3.3");
            if (File.Exists(extractTo))
                return;
            var file = File.Open(extractTo, FileMode.OpenOrCreate);
            var llvm34 = Assembly.GetExecutingAssembly().GetManifestResourceStream("CudaSharp.LLVM-3.4.dll");
            if (llvm34 == null)
                throw new Exception("Could not extract LLVM-3.4.dll");
            llvm34.CopyTo(file);
            file.Close();
            PInvoke.LoadLibrary(extractTo);
        }

        public static OpCode[] UnsupportedInstructions
        {
            get { return Translator.UnsupportedInstructions; }
        }

        public static byte[] Translate(Action method) { return Translate(method.Method); }
        public static byte[] Translate<T1>(Action<T1> method) { return Translate(method.Method); }
        public static byte[] Translate<T1, T2>(Action<T1, T2> method) { return Translate(method.Method); }
        public static byte[] Translate<T1, T2, T3>(Action<T1, T2, T3> method) { return Translate(method.Method); }
        public static byte[] Translate<T1, T2, T3, T4>(Action<T1, T2, T3, T4> method) { return Translate(method.Method); }
        public static byte[] Translate<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> method) { return Translate(method.Method); }
        public static byte[] Translate<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> method) { return Translate(method.Method); }

        public static byte[] Translate(MethodInfo method)
        {
            var module = Translator.Translate(Context.Global, method);

            return PInvokeHelper.EmitInMemory(module);

            //if (PInvoke.LLVMWriteBitcodeToFile(module, "fzoo.bc") != 0)
            //    throw new Exception("LLVMWriteBitcodeToFile returned nonzero");
            //
            //const string outfile = "fzoo.ptx";
            //InvokeLlc("fzoo.bc", outfile);
            //return outfile;
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