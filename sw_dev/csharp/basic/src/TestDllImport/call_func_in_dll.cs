using System.Runtime.InteropServices;
using System;

class call_func_in_dll
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct struct_in_dll
    {
        public Int32 count_;
        public IntPtr data_;
    }

    [DllImport("dll_func.dll")]
    private static extern int func_in_dll(int i, [MarshalAs(UnmanagedType.LPArray)] byte [] str, ref struct_in_dll data);

    public static void Main()
    {
        byte [] str = new byte [21];

        struct_in_dll data = new struct_in_dll();
        data.count_ = 5;
        int [] ia = new int [5];
        ia[0] = 2; ia[1] = 3; ia[2] = 5; ia[3] = 8; ia[4] = 13;

        GCHandle gch = GCHandle.Alloc(ia);
        data.data_ = Marshal.UnsafeAddrOfPinnedArrayElement(ia, 0);

        int ret = func_in_dll(5, str, ref data);

        Console.WriteLine("Return Value: " + ret);
        Console.WriteLine("String filled in DLL: " + System.Text.Encoding.ASCII.GetString(str));
    }
}