using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;

namespace TestGuiComponent
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void mainWindow_Loaded(object sender, RoutedEventArgs e)
        {
        }

        private void openFileDialog_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.OpenFileDialog dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.FileName = "Noname";  // default file name
            dlg.DefaultExt = ".txt";  // default file extension
            dlg.Filter = "Text files (.txt)|*.txt"; // filter files by extension

            // show open file dialog box
            Nullable<bool> result = dlg.ShowDialog();
            if (true == result)
            {
                // open document
                string filename = dlg.FileName;
            }
        }

        private void saveFileDialog_Click(object sender, RoutedEventArgs e)
        {
            Microsoft.Win32.SaveFileDialog dlg = new Microsoft.Win32.SaveFileDialog();
            dlg.FileName = "Noname";  // default file name
            dlg.DefaultExt = ".txt";  // default file extension
            dlg.Filter = "Text files (.txt)|*.txt"; // filter files by extension

            // show save file dialog box
            Nullable<bool> result = dlg.ShowDialog();
            if (true == result)
            {
                // save document
                string filename = dlg.FileName;
            }
        }

        private void printDialog_Click(object sender, RoutedEventArgs e)
        {
            System.Windows.Controls.PrintDialog dlg = new System.Windows.Controls.PrintDialog();
            dlg.PageRangeSelection = PageRangeSelection.AllPages;
            dlg.UserPageRangeEnabled = true;

            // show save file dialog box
            Nullable<bool> result = dlg.ShowDialog();
            if (result == true)
            {
                // print document
            }
        }
    }
}
