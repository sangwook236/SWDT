#include <iostream>

int wmain()
{
	void matrix_operation();
	void matrix_function();
	void vector_operation();
	void vector_function();
	void lu();
	void cholesky();
	void qr();
	void eigen();
	void svd();

	//matrix_operation();
	//matrix_function();
	//vector_operation();
	vector_function();
	//lu();
	//cholesky();
	//qr();
	//eigen();
	//svd();

	std::wcout << L"done !!!" << std::endl;
	std::wcout.flush();
	std::wcin.get();

	return 0;
}
