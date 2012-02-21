package inAction.chapter2;

import org.eclipse.jface.window.*;
import org.eclipse.swt.*;
import org.eclipse.swt.widgets.*;

public class HelloSWT_JFace extends ApplicationWindow {

	public HelloSWT_JFace() {
		super(null);
	}
	
	protected Control createContents(Composite parent)
	{
		Text helloText = new Text(parent, SWT.CENTER);
		helloText.setText("Hello SWT and JFace!");
		parent.pack();
		return parent;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		HelloSWT_JFace awin = new HelloSWT_JFace();
		awin.setBlockOnOpen(true);
		awin.open();
		Display.getCurrent().dispose();
	}

}
