/*
 * main.cpp
 *
 *  Created on: 2009. 8. 22
 *      Author: sangwook
 */

#include "eclipse_lib/Api.h"
#include "eclipse_shared/Api.h"
#include <iostream>

int main()
{
	printHello();
	std::cout << ' ';
	printWorld();
	std::cout << " !!!" << std::endl;

	return 0;
}
