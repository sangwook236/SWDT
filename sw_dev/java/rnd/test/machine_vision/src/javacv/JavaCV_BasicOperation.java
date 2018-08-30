package javacv;

import static org.bytedeco.javacpp.opencv_core.*;
//import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import org.bytedeco.javacv.Java2DFrameUtils;
import javax.swing.*;
import java.awt.*;

public class JavaCV_BasicOperation {

	public static void run(String[] args)
	{
		final String img_filename = "./data/machine_vision/beach.jpg";

		// Load image.
        final Mat img = imread(img_filename);
        if (null == img) {
            System.err.println("image file not found: " + img_filename);
            return;
        }
	
        // Flip upside down.
        flip(img, img, 0);
        // Swap red and blue channels.
        cvtColor(img, img, CV_BGR2RGB);

        // Set-up GUI.
        final JLabel imageView = new JLabel();
        imageView.setIcon(new ImageIcon(Java2DFrameUtils.toBufferedImage(img)));

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
