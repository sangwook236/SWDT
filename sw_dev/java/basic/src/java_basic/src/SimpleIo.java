import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class SimpleIo {
	
	static void runAll()
	{
/*
	    byte [] bytes = new byte[100];
	    try
	    {
			System.in.read(bytes);
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		final int sz = bytes.length;
		final String chars = new String(bytes);
		System.out.println("size: " + sz + ", data: " + chars);
*/
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		String s;
		try
		{
			while ((s = br.readLine()) != null)
			{
/*
				try
				{
					Integer i = Integer.parseInt(s);
				}
				catch (NumberFormatException e)
				{
				}
*/				
				System.out.println("size: " + s.length() + ", data: " + s);
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

}
