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
    public static class Gpu
    {
        [AttributeUsage(AttributeTargets.Method)]
        public class BuiltinAttribute : Attribute
        {
            private readonly string _intrinsic;

            public BuiltinAttribute(string intrinsic)
            {
                _intrinsic = intrinsic;
            }

            public string Intrinsic
            {
                get { return _intrinsic; }
            }
        }

        private static Exception Exception { get { return new Exception("Cannot use methods from the Gpu class on the CPU"); } }

        [Builtin("llvm.nvvm.read.ptx.sreg.tid.x")]
        public static int ThreadX() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.tid.y")]
        public static int ThreadY() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.tid.z")]
        public static int ThreadZ() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ctaid.x")]
        public static int BlockX() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ctaid.y")]
        public static int BlockY() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ctaid.z")]
        public static int BlockZ() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ntid.x")]
        public static int ThreadDimX() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ntid.y")]
        public static int ThreadDimY() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.ntid.z")]
        public static int ThreadDimZ() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.nctaid.x")]
        public static int BlockDimX() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.nctaid.y")]
        public static int BlockDimY() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.nctaid.z")]
        public static int BlockDimZ() { throw Exception; }

        [Builtin("llvm.nvvm.read.ptx.sreg.warpsize")]
        public static int WarpSize() { throw Exception; }

        [Builtin("llvm.nvvm.barrier0")]
        public static void Barrier() { throw Exception; }
    }

    static class Translator
    {
        public static Module Translate(Context context, params MethodInfo[] methods)
        {
            var module = new Module("Module", context);

            if (Environment.Is64BitOperatingSystem)
            {
                PInvoke.LLVMSetTarget(module, "nvptx64-nvidia-cuda");
                PInvoke.LLVMSetDataLayout(module, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            }
            else
            {
                PInvoke.LLVMSetTarget(module, "nvptx-nvidia-cuda");
                PInvoke.LLVMSetDataLayout(module, "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
            }

            foreach (var method in methods)
                Translate(context, module, method);
            return module;
        }

        public static void Translate(Context context, Module module, MethodInfo method)
        {
            var function = EmitFunction(context, module, method);

            var metadataArgs = new[]
            {
                function, PInvoke.LLVMMDStringInContext(context, method.Name),
                IntegerType.GetInt32(context).Constant(1, true)
            };
            var metadata = PInvoke.LLVMMDNodeInContext(context, metadataArgs);
            PInvoke.LLVMAddNamedMetadataOperand(module, "nvvm.annotations", metadata);
        }

        private static Function EmitFunction(Context context, Module module, MethodInfo method)
        {
            var funcType = new FunctionType(ConvertType(context, method.ReturnType), AnalyzeArguments(context, method.GetParameters()));

            var intrinsic = method.GetCustomAttribute<Gpu.BuiltinAttribute>();
            if (intrinsic != null)
            {
                var name = intrinsic.Intrinsic;
                var preExisting = module.GetFunction(name);
                if (preExisting != null)
                    return preExisting;
                return module.CreateFunction(name, funcType);
            }

            var function = module.CreateFunction(method.Name, funcType);

            var block = new Block("entry", context, function);
            var writer = new InstructionBuilder(context, block);

            var opcodes = method.Decompile().ToList();
            FindBranchTargets(opcodes, context, function);

            var body = method.GetMethodBody();
            var efo = new EmitFuncObj(context, module, function, writer, null, new Stack<Value>(),
                body == null ? null : new Value[body.LocalVariables.Count], new Value[method.GetParameters().Length]);

            foreach (var opcode in opcodes)
            {
                if (EmitFunctions.ContainsKey(opcode.Opcode) == false)
                    throw new Exception("Unsupported CIL instruction " + opcode.Opcode);
                var func = EmitFunctions[opcode.Opcode];
                efo.Argument = opcode.Parameter;
                func(efo);
            }

            return function;
        }

        private static Type[] AnalyzeArguments(Context context, IEnumerable<ParameterInfo> parameters)
        {
            return parameters.Select(p => ConvertType(context, p.ParameterType)).ToArray();
        }

        private static Type ConvertType(Context context, System.Type type)
        {
            if (type == typeof(void))
                return Type.GetVoid(context);
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

        class EmitFuncObj
        {
            public InstructionBuilder Builder { get; set; }
            public object Argument { get; set; }
            public Context Context { get; private set; }
            public Module Module { get; set; }
            public Function Function { get; private set; }
            public Stack<Value> Stack { get; private set; }
            public Value[] Locals { get; private set; }
            public Value[] Parameters { get; private set; }

            public EmitFuncObj(Context context, Module module, Function function, InstructionBuilder instructionBuilder, object argument, Stack<Value> stack, Value[] locals, Value[] parameters)
            {
                Context = context;
                Module = module;
                Function = function;
                Builder = instructionBuilder;
                Argument = argument;
                Stack = stack;
                Locals = locals;
                Parameters = parameters;
            }
        }

        private static void FindBranchTargets(IList<OpCodeInstruction> opCodes, Context context, Function function)
        {
            for (var i = 0; i < opCodes.Count; i++)
            {
                var op = opCodes[i];
                var opcode = op.Opcode;
                switch (opcode.FlowControl)
                {
                    case FlowControl.Branch:
                    case FlowControl.Cond_Branch:
                        break;
                    default:
                        continue;
                }

                var target = Convert.ToInt32(op.Parameter);
                target += (int)opCodes[i + 1].InstructionStart;

                var insert = 0;
                while (opCodes[insert].InstructionStart != target)
                    insert++;

                var contBlock = new Block("", context, function);
                Block block;
                if (opCodes[insert].Opcode == OpCodes.Nop && opCodes[insert].Parameter != null)
                    block = (Block)opCodes[insert].Parameter;
                else
                {
                    opCodes.Insert(insert, new OpCodeInstruction(target, OpCodes.Nop, block = new Block("", context, function)));
                    if (insert < i)
                        i++;
                }
                opCodes[i] = new OpCodeInstruction(op.InstructionStart, op.Opcode, Tuple.Create(contBlock, block));
            }
        }

        private delegate void EmitFunc(EmitFuncObj arg);

        private static readonly Dictionary<OpCode, EmitFunc> EmitFunctions = new Dictionary<OpCode, EmitFunc>
        {
            {OpCodes.Nop, Nop},
            {OpCodes.Pop, _ => _.Stack.Pop()},
            {OpCodes.Dup, _ => _.Stack.Push(_.Stack.Peek())},
            {OpCodes.Ret, _ => { if (_.Stack.Count == 0) _.Builder.Return(); else _.Builder.Return(_.Stack.Pop()); _.Builder = null; }},
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
            {OpCodes.Ldc_I8, _ => _.Stack.Push(IntegerType.Get(_.Context, 64).Constant(ulong.MaxValue, true))},
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
            {OpCodes.Ldind_I, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_I1, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_I2, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_I4, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_I8, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_R4, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_R8, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_U1, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_U2, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_U4, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Ldind_Ref, _ => _.Stack.Push(_.Builder.Load(_.Stack.Pop()))},
            {OpCodes.Stind_I, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_I1, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_I2, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_I4, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_I8, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_R4, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_R8, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Stind_Ref, _ => _.Builder.Store(_.Stack.Pop(), _.Stack.Pop())},
            {OpCodes.Conv_I1, _ => ConvertNum(_, IntegerType.Get(_.Context, 8 ), true)},
            {OpCodes.Conv_I2, _ => ConvertNum(_, IntegerType.Get(_.Context, 16), true)},
            {OpCodes.Conv_I4, _ => ConvertNum(_, IntegerType.Get(_.Context, 32), true)},
            {OpCodes.Conv_I8, _ => ConvertNum(_, IntegerType.Get(_.Context, 64), true)},
            {OpCodes.Conv_U1, _ => ConvertNum(_, IntegerType.Get(_.Context, 8 ), false)},
            {OpCodes.Conv_U2, _ => ConvertNum(_, IntegerType.Get(_.Context, 16), false)},
            {OpCodes.Conv_U4, _ => ConvertNum(_, IntegerType.Get(_.Context, 32), false)},
            {OpCodes.Conv_U8, _ => ConvertNum(_, IntegerType.Get(_.Context, 64), false)},
            {OpCodes.Conv_R4, _ => ConvertNum(_, FloatType.Get(_.Context, 32), true)},
            {OpCodes.Conv_R8, _ => ConvertNum(_, FloatType.Get(_.Context, 64), true)},
            {OpCodes.Neg, _ => _.Stack.Push(_.Builder.Negate(_.Stack.Pop()))},
            {OpCodes.Not, _ => _.Stack.Push(_.Builder.Not(_.Stack.Pop()))},
            {OpCodes.Add, _ => _.Stack.Push(_.Builder.Add(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Sub, _ => _.Stack.Push(_.Builder.Subtract(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Mul, _ => _.Stack.Push(_.Builder.Multiply(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Div, _ => _.Stack.Push(_.Builder.Divide(true, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Div_Un, _ => _.Stack.Push(_.Builder.Divide(false, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Rem, _ => _.Stack.Push(_.Builder.Reminder(true, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Rem_Un, _ => _.Stack.Push(_.Builder.Reminder(false, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.And, _ => _.Stack.Push(_.Builder.And(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Or, _ => _.Stack.Push(_.Builder.Or(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Xor, _ => _.Stack.Push(_.Builder.Xor(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Shl, _ => _.Stack.Push(_.Builder.ShiftLeft(_.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Shr, _ => _.Stack.Push(_.Builder.ShiftRight(true, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Shr_Un, _ => _.Stack.Push(_.Builder.ShiftRight(false, _.Stack.Pop(), _.Stack.Pop()))},
            {OpCodes.Ceq, Ceq},
            {OpCodes.Cgt, Cgt},
            {OpCodes.Cgt_Un, CgtUn},
            {OpCodes.Clt, Clt},
            {OpCodes.Clt_Un, CltUn},
            {OpCodes.Ldloc, _ => LdVar(_, _.Locals, Convert.ToInt32(_.Argument), false)},
            {OpCodes.Ldloc_S, _ => LdVar(_, _.Locals, Convert.ToInt32(_.Argument), false)},
            {OpCodes.Ldloc_0, _ => LdVar(_, _.Locals, 0, false)},
            {OpCodes.Ldloc_1, _ => LdVar(_, _.Locals, 1, false)},
            {OpCodes.Ldloc_2, _ => LdVar(_, _.Locals, 2, false)},
            {OpCodes.Ldloc_3, _ => LdVar(_, _.Locals, 3, false)},
            {OpCodes.Stloc, _ => StVar(_, _.Locals, Convert.ToInt32(_.Argument))},
            {OpCodes.Stloc_S, _ => StVar(_, _.Locals, Convert.ToInt32(_.Argument))},
            {OpCodes.Stloc_0, _ => StVar(_, _.Locals, 0)},
            {OpCodes.Stloc_1, _ => StVar(_, _.Locals, 1)},
            {OpCodes.Stloc_2, _ => StVar(_, _.Locals, 2)},
            {OpCodes.Stloc_3, _ => StVar(_, _.Locals, 3)},
            {OpCodes.Ldarg, _ => LdVar(_, _.Parameters, Convert.ToInt32(_.Argument), true)},
            {OpCodes.Ldarg_S, _ => LdVar(_, _.Parameters, Convert.ToInt32(_.Argument), true)},
            {OpCodes.Ldarg_0, _ => LdVar(_, _.Parameters, 0, true)},
            {OpCodes.Ldarg_1, _ => LdVar(_, _.Parameters, 1, true)},
            {OpCodes.Ldarg_2, _ => LdVar(_, _.Parameters, 2, true)},
            {OpCodes.Ldarg_3, _ => LdVar(_, _.Parameters, 3, true)},
            {OpCodes.Starg, _ => StVar(_, _.Parameters, Convert.ToInt32(_.Argument))},
            {OpCodes.Starg_S, _ => StVar(_, _.Parameters, Convert.ToInt32(_.Argument))},
            {OpCodes.Br, Br},
            {OpCodes.Br_S, Br},
            {OpCodes.Brtrue, _ => BrCond(_, true)},
            {OpCodes.Brtrue_S, _ => BrCond(_, true)},
            {OpCodes.Brfalse, _ => BrCond(_, false)},
            {OpCodes.Brfalse_S, _ => BrCond(_, false)},
            {OpCodes.Beq, _ => {Ceq(_);BrCond(_, true);}},
            {OpCodes.Beq_S, _ => {Ceq(_);BrCond(_, true);}},
            {OpCodes.Bne_Un, _ => {Ceq(_);BrCond(_, false);}},
            {OpCodes.Bne_Un_S, _ => {Ceq(_);BrCond(_, false);}},
            {OpCodes.Ble, _ => {Cle(_);BrCond(_, true);}},
            {OpCodes.Ble_S, _ => {Cle(_);BrCond(_, true);}},
            {OpCodes.Ble_Un, _ => {CleUn(_);BrCond(_, true);}},
            {OpCodes.Ble_Un_S, _ => {CleUn(_);BrCond(_, true);}},
            {OpCodes.Blt, _ => {Clt(_);BrCond(_, true);}},
            {OpCodes.Blt_S, _ => {Clt(_);BrCond(_, true);}},
            {OpCodes.Blt_Un, _ => {CltUn(_);BrCond(_, true);}},
            {OpCodes.Blt_Un_S, _ => {CltUn(_);BrCond(_, true);}},
            {OpCodes.Bge, _ => {Cge(_);BrCond(_, true);}},
            {OpCodes.Bge_S, _ => {Cge(_);BrCond(_, true);}},
            {OpCodes.Bge_Un, _ => {CgeUn(_);BrCond(_, true);}},
            {OpCodes.Bge_Un_S, _ => {CgeUn(_);BrCond(_, true);}},
            {OpCodes.Bgt, _ => {Cgt(_);BrCond(_, true);}},
            {OpCodes.Bgt_S, _ => {Cgt(_);BrCond(_, true);}},
            {OpCodes.Bgt_Un, _ => {CgtUn(_);BrCond(_, true);}},
            {OpCodes.Bgt_Un_S, _ => {CgtUn(_);BrCond(_, true);}},
            {OpCodes.Tailcall, Call},
            {OpCodes.Call, Call},
        };

        private static void Call(EmitFuncObj _)
        {
            var method = (MethodInfo)_.Argument;
            var args = method.GetParameters().Select(x => _.Stack.Pop()).Reverse().ToArray();
            var result = _.Builder.Call(EmitFunction(_.Context, _.Module, method), args);
            if (result.Type.StructuralEquals(Type.GetVoid(_.Context)) == false)
                _.Stack.Push(result);
        }

        private static void Ceq(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.Equal, PopNoBool(_), PopNoBool(_)));
        }

        private static void Cgt(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.SignedGreater, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void CgtUn(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.UnsignedGreater, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void Cge(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.SignedGreaterEqual, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void CgeUn(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.UnsignedGreaterEqual, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void Clt(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.SignedLess, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void CltUn(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.UnsignedLess, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void Cle(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.SignedLessEqual, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static void CleUn(EmitFuncObj _)
        {
            _.Stack.Push(_.Builder.Compare(IntegerComparison.UnsignedLessEqual, _.Stack.Pop(), _.Stack.Pop()));
        }

        private static readonly OpCode[] UnsupportedInstructionsArr = typeof(OpCodes).GetFields().Select(f => (OpCode)f.GetValue(null)).Where(op => !EmitFunctions.ContainsKey(op)).ToArray();
        public static OpCode[] UnsupportedInstructions
        {
            get { return UnsupportedInstructionsArr; }
        }

        private static Value PopNoBool(EmitFuncObj _)
        {
            var popped = _.Stack.Pop();
            if (popped.Type.StructuralEquals(IntegerType.Get(_.Context, 1)))
                popped = _.Builder.ZeroExtend(popped, IntegerType.GetInt32(_.Context));
            return popped;
        }

        private static void Nop(EmitFuncObj _)
        {
            var block = (Block)_.Argument;
            if (block == null)
                return;
            if (_.Builder != null)
                _.Builder.GoTo(block);
            _.Builder = new InstructionBuilder(_.Context, block);
        }

        private static void Br(EmitFuncObj _)
        {
            _.Builder.GoTo((Block)_.Argument);
            _.Builder = null;
        }

        private static void BrCond(EmitFuncObj _, bool isTrue)
        {
            var tuple = (Tuple<Block, Block>)_.Argument;
            var cont = tuple.Item1;
            var target = tuple.Item2;
            _.Builder.If(_.Stack.Pop(), isTrue ? target : cont, isTrue ? cont : target);
            _.Builder = new InstructionBuilder(_.Context, cont);
        }

        private static void LdVar(EmitFuncObj _, Value[] values, int index, bool isParameter)
        {
            if (values[index] == null)
            {
                if (!isParameter)
                    throw new Exception("Uninitialized variable at index " + index);
                var arg = _.Function[index];
                values[index] = _.Builder.StackAlloc(_.Function[index].Type);
                _.Builder.Store(arg, values[index]);
                _.Stack.Push(arg);
            }
            else
            {
                var load = _.Builder.Load(values[index]);
                _.Stack.Push(load);
            }
        }

        private static void StVar(EmitFuncObj _, Value[] values, int index)
        {
            var pop = _.Stack.Pop();
            if (values[index] == null)
                values[index] = _.Builder.StackAlloc(pop.Type);
            _.Builder.Store(pop, values[index]);
        }

        private static void StElem(EmitFuncObj _)
        {
            var value = _.Stack.Pop();
            var index = _.Stack.Pop();
            var array = _.Stack.Pop();
            var idx = _.Builder.Element(array, new[] { index });
            _.Builder.Store(value, idx);
        }

        private static void LdElem(EmitFuncObj _)
        {
            var index = _.Stack.Pop();
            var array = _.Stack.Pop();
            var idx = _.Builder.Element(array, new[] { index });
            var load = _.Builder.Load(idx);
            _.Stack.Push(load);
        }

        private static void ConvertNum(EmitFuncObj _, Type target, bool integerSignedness)
        {
            //var value = _.Stack.Pop();
            //var valueType = value.Type;
            //if (valueType is IntegerType && target is FloatType)
            //{
            //    value = _.Builder.
            //}
        }
    }
}