using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaSharp
{
    static class Extensions
    {
        public static int IndexOf<T>(this IEnumerable<T> enumerable, Func<T, bool> selector)
        {
            return enumerable.Select((e, i) => new {e, i}).First(pair => selector(pair.e)).i;
        }
    }
}