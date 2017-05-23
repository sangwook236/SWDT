package java_beginning_2;

import java.io.*;

public class JavaTest1
{
	public static void main(String[] args)
	{
/*
	    byte [] bytes = new byte[100];
	    try {
			System.in.read(bytes);
		} catch (IOException ex) {
			// TODO Auto-generated catch block
			ex.printStackTrace();
		}
		final int sz = bytes.length;
		final String chars = new String(bytes);
		System.out.println("size: " + sz + ", data: " + chars);
*/
		BufferedReader br
		= new BufferedReader(new InputStreamReader(System.in));
		String s;
		try {
			while ((s = br.readLine()) != null)
				System.out.println("size: " + s.length() + ", data: " + s);
		} catch (IOException ex) {
			// TODO Auto-generated catch block
			ex.printStackTrace();
		}
	}
}
