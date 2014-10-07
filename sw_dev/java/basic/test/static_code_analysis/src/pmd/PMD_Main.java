package pmd;

import java.util.*;

public class PMD_Main {

	public static void run(String[] args) {
		
		System.out.println("중복 소스코드 검출");
		String url;
		if (args.length == 0)
		{
			url = "http://www.test.com/test";
		}
		else
		{
			url = args[0];
		}
		System.out.println("중복 소스코드 검출");
		
		if (true)
		{
			int temp = 1;
			temp = 2;
			temp = 3;
			temp = 4;
			temp = 5;
			temp = 6;
			temp = 7;
			temp = 8;
			temp = 9;
		}
		
		StringTokenizer st = new StringTokenizer(url, ":/.~", false);
		while (st.hasMoreTokens())
		{
			String token = st.nextToken();
			System.out.println("token => " + token);
		}

		if (true)
		{
			int temp = 1;
			temp = 2;
			temp = 3;
			temp = 4;
			temp = 5;
			temp = 6;
			temp = 7;
			temp = 8;
			temp = 9;
		}

	}

}
