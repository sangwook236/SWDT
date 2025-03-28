package apache_pivot;

import org.apache.pivot.beans.BXMLSerializer;
import org.apache.pivot.collections.Map;
import org.apache.pivot.wtk.Application;
import org.apache.pivot.wtk.Display;
import org.apache.pivot.wtk.Window;
import org.apache.pivot.wtk.DesktopApplicationContext;

public class LinkButtonTest implements Application
{
	public static void run(String[] args)
	{
	    DesktopApplicationContext.main(LinkButtonTest.class, args);
	}

	@Override
    public void startup(Display display, Map<String, String> properties) throws Exception
    {
        BXMLSerializer bxmlSerializer = new BXMLSerializer();
        window_ = (Window)bxmlSerializer.readObject(LinkButtonTest.class, "bxml/link_button.bxml");
        window_.open(display);
    }
 
    @Override
    public boolean shutdown(boolean optional)
    {
        if (null != window_)
        {
            window_.close();
        }
 
        return false;
    }
 
    @Override
    public void suspend()
    {
    }
 
    @Override
    public void resume()
    {
    }

    private Window window_ = null;
}
