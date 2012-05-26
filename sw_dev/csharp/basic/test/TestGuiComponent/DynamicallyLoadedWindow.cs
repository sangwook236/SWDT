using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Markup;
using System.IO;

namespace TestGuiComponent
{
    /// <summary>
    /// </summary>
    public partial class DynamicallyLoadedWindow : Window
    {
        public DynamicallyLoadedWindow()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            // configure this window
            this.Width = 640;
            this.Height = 480;
            //this.Left = this.Top = 100;
            this.Title = "Dynamically Loaded XAML";

            // get the XAML content from an external file
            FileStream stream = new FileStream("..\\data\\xaml\\DynamicallyLoadedWindow.xml", FileMode.Open);
            DependencyObject rootElement = (DependencyObject)XamlReader.Load(stream);
            this.Content = rootElement;

            // find the control with the appropriate name
#if true
            ticketButton_ = (Button)LogicalTreeHelper.FindLogicalNode(rootElement, "ticketButton");
#else
            FrameworkElement frameworkElement = (FrameworkElement)rootElement;
            ticketButton_ = (Button)frameworkElement.FindName("ticketButton");
#endif

            // wire up the event handler
            ticketButton_.Click += ticketButton_Click;
        }

        private void ticketButton_Click(object sender, RoutedEventArgs e)
        {
            this.Cursor = Cursors.Wait;

            System.Threading.Thread.Sleep(TimeSpan.FromSeconds(1));

            this.Cursor = null;
        }

        private Button ticketButton_;
    }
}
