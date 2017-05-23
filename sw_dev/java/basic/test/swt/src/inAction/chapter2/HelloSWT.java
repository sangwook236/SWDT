package inAction.chapter2;
import org.eclipse.swt.*;
import org.eclipse.swt.widgets.*;

public class HelloSWT {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Display display = new Display();
		Shell shell = new Shell(display);

		Text helloText = new Text(shell, SWT.CENTER);
		helloText.setText("Hellow SWT!");
		helloText.pack();

		shell.pack();
		shell.open();
		while (!shell.isDisposed())
		{
			if (!display.readAndDispatch())
				display.sleep();
			display.dispose();
		}
	}

}
