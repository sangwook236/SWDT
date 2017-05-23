package runtime;

public class MemoryCheck {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Runtime runtime = Runtime.getRuntime();

		final long freeMem = runtime.freeMemory();
		final long totalMem = runtime.totalMemory();
		final long maxMem = runtime.maxMemory();
		//sizeOf();

		System.out.println(freeMem);
		System.out.println(totalMem);
		System.out.println(maxMem);
	}

}
