#if !defined(__CONSOLE_WINDOW__H_)
#define __CONSOLE_WINDOW__H_ 1


class ConsoleWindow
{
private:
	ConsoleWindow();
public:
	~ConsoleWindow();

public:
	static ConsoleWindow & getInstance();

	static void initialize();
	static void finalize();

	bool isValid() const  {  return isValid_;  }

private:
	bool isValid_;
};


#endif  // __CONSOLE_WINDOW__H_
