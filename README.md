CudaSharp
=========

A library to make C# run on CUDA-enabled GPUs

This library is not complete and is not intended for general use. Testers are, however, very welcome.

C# is executed on the GPU like so:

* Your C# compiler translates C# to .net CIL

* Calling the CudaSharp.Translate method reads the CIL, translates to LLVM IR, and compiles to NVIDIA PTX

* Use your GPU library of choice (for example, ManagedCuda) to read in the PTX file and execute it on the GPU

For an example usage, see CudaSharpTest/Program.cs
