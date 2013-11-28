package javacv;

import static com.googlecode.javacv.cpp.opencv_core.IplImage;
import static com.googlecode.javacv.cpp.opencv_core.cvFlip;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_BGR2RGB;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import javax.swing.*;
import java.awt.*;

public class JavaCV_BasicOperation {

	public static void run(String[] args)
	{
		final String img_filename = "./data/machine_vision/beach.jpg";

		// Load image.
        final IplImage img = cvLoadImage(img_filename);
        if (null == img) {
            System.err.println("image file not found: " + img_filename);
            return;
        }
	
        // Flip upside down.
        cvFlip(img, img, 0);
        // Swap red and blue channels.
        cvCvtColor(img, img, CV_BGR2RGB);

        // Set-up GUI.
        final JLabel imageView = new JLabel();
        imageView.setIcon(new ImageIcon(img.getBufferedImage()));

        final JScrollPane imageScrollPane = new JScrollPane(imageView);
        imageScrollPane.setPreferredSize(new Dimension(640, 480));
        
        final JFrame frame = new JFrame();
        frame.add(imageScrollPane, BorderLayout.CENTER);

        frame.pack();
        // Mark for display in the center of the screen.
        frame.setLocationRelativeTo(null);
        // Exit application when frame is closed.
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setVisible(true);
	}

}
