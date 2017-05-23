/**
 * 
 */
package inAction.chapter2;

import org.eclipse.swt.widgets.*;
import org.eclipse.jface.window.*;

/**
 * @author sangwook
 *
 */
public class WidgetWindow extends ApplicationWindow {

	public WidgetWindow()
	{
		super(null);
	}

	protected Control createContents(Composite parent)
	{
		getShell().setText("Widget Window");
		parent.setSize(400, 250);
		return parent;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		WidgetWindow wwin = new WidgetWindow();
		wwin.setBlockOnOpen(true);
		wwin.open();
		Display.getCurrent().dispose();
	}

}
