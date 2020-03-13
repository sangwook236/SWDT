package pdfbox;

import java.io.File;
import java.io.IOException;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.encryption.AccessPermission;
import org.apache.pdfbox.text.PDFTextStripper;

public class PDFBox_Main {

	public static void run(String[] args)
	{
		try
		{
			extractTextSimple();
		}
		catch (IOException ex)
		{
		}
	}

	// REF [file] >> ${PDFBox_HOME}/examples/util/ExtractTextSimple.java
	static void extractTextSimple() throws IOException
	{
		String pdf_filepath = "./sample.pdf";

		System.out.println("=============================0");
		try (PDDocument document = PDDocument.load(new File(pdf_filepath)))
		{
			System.out.println("=============================");

			AccessPermission ap = document.getCurrentAccessPermission();
			if (!ap.canExtractContent())
			{
				throw new IOException("You do not have permission to extract text");
			}

			PDFTextStripper stripper = new PDFTextStripper();

			// This example uses sorting, but in some cases it is more useful to switch it off,
			// e.g. in some files with columns where the PDF content stream respects the
			// column order.
			stripper.setSortByPosition(true);

			for (int p = 1; p <= document.getNumberOfPages(); ++p)
			{
				// Set the page interval to extract. If you don't, then all pages would be extracted.
				stripper.setStartPage(p);
				stripper.setEndPage(p);

				// Let the magic happen.
				String text = stripper.getText(document);

				// Do some nice output with a header.
				String pageStr = String.format("page %d:", p);
				System.out.println(pageStr);
				for (int i = 0; i < pageStr.length(); ++i)
				{
					System.out.print("-");
				}
				System.out.println();
				System.out.println(text.trim());
				System.out.println();

				// If the extracted text is empty or gibberish, please try extracting text
				// with Adobe Reader first before asking for help. Also read the FAQ
				// on the website: 
				// https://pdfbox.apache.org/2.0/faq.html#text-extraction
			}
		}
		System.out.println("=============================1");
	}

}
