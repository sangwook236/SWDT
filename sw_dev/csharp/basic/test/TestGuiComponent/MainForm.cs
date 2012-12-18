using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace TestGuiComponent
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void openFileDialogButton_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();

            //openFileDialog.InitialDirectory = "c:\\";
            openFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            openFileDialog.FilterIndex = 2;
            openFileDialog.RestoreDirectory = true;

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    Stream myStream = openFileDialog.OpenFile();
                    if (null != myStream)
                    {
                        using (myStream)
                        {
                            // Insert code to read from the stream here.
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error: Could not read a file. Original error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void saveFileDialogButton_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();

            //openFileDialog.InitialDirectory = "c:\\";
            saveFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            saveFileDialog.FilterIndex = 2;
            saveFileDialog.RestoreDirectory = true;

            if (saveFileDialog.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    Stream myStream = saveFileDialog.OpenFile();
                    if (null != myStream)
                    {
                        using (myStream)
                        {
                            // Insert code to save to the stream here.
                        }
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error: Could not save the file. Original error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void printDialogButton_Click(object sender, EventArgs e)
        {
            PrintDialog printDialog = new PrintDialog();

            // Allow the user to choose the page range he or she would like to print.
            printDialog.AllowSomePages = true;

            // Show the help button.
            printDialog.ShowHelp = true;

            // Set the Document property to the PrintDocument for which the PrintPage Event has been handled.
            // To display the dialog, either this property or the PrinterSettings property must be set 
            printDialog.Document = documentToPrint_;

            DialogResult result = printDialog.ShowDialog();

            // If the result is OK then print the documentToPrint_.
            if (result == DialogResult.OK)
            {
                documentToPrint_.PrintPage += documentToPrint_PrintPage;
                documentToPrint_.Print();
            }
        }

        // The PrintDialog will print the documentToPrint_ by handling the documentToPrint_'s PrintPage event.
        private void documentToPrint_PrintPage(object sender, System.Drawing.Printing.PrintPageEventArgs e)
        {
            // Insert code to render the page here.
            // This code will be called when the control is drawn.

            // The following code will render a simple message on the printed documentToPrint_.
            string text = "In document_PrintPage method.";
            System.Drawing.Font printFont = new System.Drawing.Font("Arial", 35, System.Drawing.FontStyle.Regular);

            // Draw the content.
            e.Graphics.DrawString(text, printFont, System.Drawing.Brushes.Black, 10, 10);
        }

        // Declare the PrintDocument object.
        private System.Drawing.Printing.PrintDocument documentToPrint_ = new System.Drawing.Printing.PrintDocument();
    }
}
