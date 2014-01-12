using System;
using System.Runtime.InteropServices;
using CC = System.Runtime.InteropServices.CallingConvention;

namespace CudaSharp
{
    static class PInvoke
    {
        const string LlvmDll = "LLVM-3.3";

        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern int LLVMWriteBitcodeToFile(IntPtr module, string filename);
        
        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern void LLVMSetTarget(IntPtr module, string triple);

        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern void LLVMSetDataLayout(IntPtr module, string triple);

        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern void LLVMAddNamedMetadataOperand(IntPtr module, string name, IntPtr value);

        // ReSharper disable once InconsistentNaming
        public static IntPtr LLVMMDNodeInContext(IntPtr context, IntPtr[] values)
        {
            return LLVMMDNodeInContext(context, values, (uint)values.Length);
        }

        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern IntPtr LLVMMDNodeInContext(IntPtr context, IntPtr[] values, uint count);

        // ReSharper disable once InconsistentNaming
        public static IntPtr LLVMMDStringInContext(IntPtr context, string str)
        {
            return LLVMMDStringInContext(context, str, (uint)str.Length);
        }

        [DllImport(LlvmDll, CallingConvention = CC.Cdecl)]
        public static extern IntPtr LLVMMDStringInContext(IntPtr context, string str, uint strLen);
    }
}
