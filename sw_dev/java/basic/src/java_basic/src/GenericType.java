
class Cell<T>
{
	Cell(T val)
	{
		value_ = val;
	}
	
	T get()
	{
		return value_;
	}
	
	void set(T val)
	{
		value_ = val;
	}
	
	T value_;
}

public class GenericType {
	
	/**
	 * @param args
	 */
	static void runAll() {
		Cell<String> cs = new Cell<String>("abc");
		System.out.println(cs.value_);
		System.out.println(cs.get());
		cs.set("def");
		System.out.println(cs.get());
	}

}
