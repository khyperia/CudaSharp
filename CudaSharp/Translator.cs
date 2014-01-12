using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using LLVM;
using Module = LLVM.Module;
using Type = LLVM.Type;

namespace CudaSharp
{
    static class Translator
    {
        public static Module Translate(Context context, MethodInfo method)
        {
            var module = new Module("Module", context);

            PInvoke.LLVMSetTarget(module, "nvptx64-nvidia-cuda");
            PInvoke.LLVMSetDataLayout(module, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");

            Translate(context, module, method);
            return module;
        }

        public static void Translate(Context context, Module module, MethodInfo method)
        {
            var funcType = new FunctionType(Type.GetVoid(context), AnalyzeArguments(context, method.GetParameters()));
            var function = module.CreateFunction(method.Name, funcType);

            var block = new Block("entry", context, function);
            var writer = new InstructionBuilder(context, block);

            var stack = new Stack<Value>();

            foreach (var opcode in method.Decompile())
            {
                if (EmitFunctions.ContainsKey(opcode.Opcode) == false)
                    throw new Exception("Unsupported CIL instruction " + opcode.Opcode);
                var func = EmitFunctions[opcode.Opcode];
                var arg = new EmitFuncStruct(context, function, writer, opcode.Parameter, stack);
                func(arg);
            }

            var metadataArgs = new[]
            {
                function, PInvoke.LLVMMDStringInContext(context, method.Name),
                IntegerType.GetInt32(context).Constant(1, true)
            };
            var metadata = PInvoke.LLVMMDNodeInContext(context, metadataArgs);
            PInvoke.LLVMAddNamedMetadataOperand(module, "nvvm.annotations", metadata);
        }

        private static Type[] AnalyzeArguments(Context context, IEnumerable<ParameterInfo> parameters)
        {
            return parameters.Select(p => ConvertType(context, p.ParameterType)).ToArray();
        }

        private static Type ConvertType(Context context, System.Type type)
        {
            if (type == typeof(bool))
                return IntegerType.Get(context, 1);
            if (type == typeof(byte))
                return IntegerType.Get(context, 8);
            if (type == typeof(short))
                return IntegerType.Get(context, 16);
            if (type == typeof(int))
                return IntegerType.GetInt32(context);
            if (type == typeof(long))
                return IntegerType.Get(context, 64);
            if (type == typeof(float))
                return FloatType.Get(context, 32);
            if (type == typeof(double))
                return FloatType.Get(context, 64);
            if (type.IsArray)
                return PointerType.Get(ConvertType(context, type.GetElementType()), 1);

            throw new Exception("Type cannot be translated to CUDA: " + type.FullName);
        }

        struct EmitFuncStruct
        {
            private readonly Context _context;
            private readonly Function _function;
            private readonly InstructionBuilder _instructionBuilder;
            private readonly object _argument;
            private readonly Stack<Value> _stack;

            public EmitFuncStruct(Context context, Function function, InstructionBuilder instructionBuilder, object argument, Stack<Value> stack)
            {
                _context = context;
                _function = function;
                _instructionBuilder = instructionBuilder;
                _argument = argument;
                _stack = stack;
            }

            public Context Context
            {
                get { return _context; }
            }

            public Function Function
            {
                get { return _function; }
            }

            public InstructionBuilder Builder
            {
                get { return _instructionBuilder; }
            }

            public object Argument
            {
                get { return _argument; }
            }

            public Stack<Value> Stack
            {
                get { return _stack; }
            }
        }

        private delegate void EmitFunc(EmitFuncStruct arg);

        private static readonly Dictionary<OpCode, EmitFunc> EmitFunctions = new Dictionary<OpCode, EmitFunc>
        {
            {OpCodes.Nop, _ => { }},
            {OpCodes.Ret, _ => { if (_.Stack.Count == 0) _.Builder.Return(); else _.Builder.Return(_.Stack.Pop());}},
            {OpCodes.Ldarg, _ => _.Stack.Push(_.Function[Convert.ToInt32(_.Argument)])},
            {OpCodes.Ldarg_S, _ => _.Stack.Push(_.Function[Convert.ToInt32(_.Argument)])},
            {OpCodes.Ldarg_0, _ => _.Stack.Push(_.Function[0])},
            {OpCodes.Ldarg_1, _ => _.Stack.Push(_.Function[1])},
            {OpCodes.Ldarg_2, _ => _.Stack.Push(_.Function[2])},
            {OpCodes.Ldarg_3, _ => _.Stack.Push(_.Function[3])},
            {OpCodes.Ldc_I4, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(Convert.ToUInt64(_.Argument), true))},
            {OpCodes.Ldc_I4_S, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(Convert.ToUInt64(_.Argument), true))},
            {OpCodes.Ldc_I4_0, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(0, true))},
            {OpCodes.Ldc_I4_1, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(1, true))},
            {OpCodes.Ldc_I4_2, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(2, true))},
            {OpCodes.Ldc_I4_3, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(3, true))},
            {OpCodes.Ldc_I4_4, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(4, true))},
            {OpCodes.Ldc_I4_5, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(5, true))},
            {OpCodes.Ldc_I4_6, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(6, true))},
            {OpCodes.Ldc_I4_7, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(7, true))},
            {OpCodes.Ldc_I4_8, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(8, true))},
            {OpCodes.Ldc_I4_M1, _ => _.Stack.Push(IntegerType.GetInt32(_.Context).Constant(ulong.MaxValue, true))},
            {OpCodes.Stelem, StElem},
            {OpCodes.Stelem_I, StElem},
            {OpCodes.Stelem_I1, StElem},
            {OpCodes.Stelem_I2, StElem},
            {OpCodes.Stelem_I4, StElem},
            {OpCodes.Stelem_I8, StElem},
            {OpCodes.Stelem_R4, StElem},
            {OpCodes.Stelem_R8, StElem},
            {OpCodes.Stelem_Ref, StElem},
            {OpCodes.Ldelem, LdElem},
            {OpCodes.Ldelem_I, LdElem},
            {OpCodes.Ldelem_I1, LdElem},
            {OpCodes.Ldelem_I2, LdElem},
            {OpCodes.Ldelem_I4, LdElem},
            {OpCodes.Ldelem_I8, LdElem},
            {OpCodes.Ldelem_R4, LdElem},
            {OpCodes.Ldelem_R8, LdElem},
            {OpCodes.Ldelem_Ref, LdElem},
            {OpCodes.Ldelem_U1, LdElem},
            {OpCodes.Ldelem_U2, LdElem},
            {OpCodes.Ldelem_U4, LdElem},
            {OpCodes.Add, _ => _.Stack.Push(_.Builder.Add(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Sub, _ => _.Stack.Push(_.Builder.Subtract(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Mul, _ => _.Stack.Push(_.Builder.Multiply(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Div, _ => _.Stack.Push(_.Builder.Divide(true, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Rem, _ => _.Stack.Push(_.Builder.Reminder(true, _.Stack.Pop(), _.Stack.Pop()))},
        };

        private static void StElem(EmitFuncStruct _)
        {
            var value = _.Stack.Pop();
            var index = _.Stack.Pop();
            var array = _.Stack.Pop();
            var idx = _.Builder.Element(array, new[] { index });
            _.Builder.Store(value, idx);
        }

        private static void LdElem(EmitFuncStruct _)
        {
            var index = _.Stack.Pop();
            var array = _.Stack.Pop();
            var idx = _.Builder.Element(array, new[] { index });
            var load = _.Builder.Load(idx);
            _.Stack.Push(load);
        }
    }
}