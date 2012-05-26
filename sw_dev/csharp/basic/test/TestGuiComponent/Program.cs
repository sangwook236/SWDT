using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.Windows.Navigation;

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
#elif false
            System.Windows.Application app = new System.Windows.Application();
            app.Run(new MyWindow());
#elif false
            System.Windows.Application app = new System.Windows.Application();
            System.Windows.Navigation.NavigationWindow window = new System.Windows.Navigation.NavigationWindow();
            MyPage page = new MyPage();
            window.Navigate(page);
            app.Run(window);
#elif true
            System.Windows.Application app = new System.Windows.Application();
            app.Run(new DynamicallyLoadedWindow());
#endif
        }
    }
}