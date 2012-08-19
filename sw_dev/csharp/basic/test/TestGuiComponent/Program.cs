using System;
using System.Collections.Generic;
using System.Linq;
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
#if true
            // set TestGuiComponent.Program as "Startup object" in this project's Property window

#if false
            // WPF: dynamically loaded window.
            System.Windows.Application app = new System.Windows.Application();
            app.Run(new DynamicallyLoadedWindow());
#elif false
            // WPF
            System.Windows.Application app = new System.Windows.Application();
            app.Run(new MyWindow());
#elif false
            // WPF
            System.Windows.Application app = new System.Windows.Application();
            System.Windows.Navigation.NavigationWindow window = new System.Windows.Navigation.NavigationWindow();
            MyPage page = new MyPage();
            window.Navigate(page);
            app.Run(window);
#else
            // Windows Forms
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
#endif

#else
            // set TestGuiComponent.App as "Startup object" in this project's Property window

            // WPF
            //System.Windows.Application app = new TestGuiComponent.App();
            //app.InitializeComponent();
            //app.Run();
#endif
        }
    }
}