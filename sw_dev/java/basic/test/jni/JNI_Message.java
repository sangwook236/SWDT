class JNI_Message
{
	native String Message(String input);
   
  // 라이브러리 적재(Load the library)  
	static
	{
		System.loadLibrary("Msg_DLL");
	}

	public static void main(String args[])
	{
		String buf;

		// 클래스 인스턴스 생성(Create class instance)
		JNI_Message myJNI = new JNI_Message();

		// 원시 메소드에 값을 주고 받음
		buf = myJNI.Message("Apple");

		System.out.print(buf); // 받은값 출력
	}
}
