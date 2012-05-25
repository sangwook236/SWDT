using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace TestGuiComponent
{
    static class Program
    {
        /// <summary>
        /// 해당 응용 프로그램의 주 진입점입니다.
        /// </summary>
        [STAThread]
        static void Main()
        {
#if false
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
#else
            System.Windows.Application app = new System.Windows.Application();

            MyWindow win = new MyWindow();

            app.Run(win);
#endif
        }
    }
}