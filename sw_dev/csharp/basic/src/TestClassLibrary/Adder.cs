using System;
using System.Collections.Generic;
using System.Text;

namespace TestClassLibrary
{
    /// <summary>
    /// Adder class
    /// </summary>
    public class Adder
    {
        /// <summary>
        /// add() static function
        /// </summary>
        /// <param name="lhs">left-hand operand</param>
        /// <param name="rhs">right-hand operand</param>
        /// <returns>result of lhs + rhs</returns>
        public static int add(int lhs, int rhs)
        {
            return lhs + rhs;
        }
    }
}
