// AddFrontConsole.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include "EventHandler.h"
#include <iostream>

CComModule _Module;

int _tmain(int argc, _TCHAR* argv[])
{
	_Module.Init(NULL, NULL);
	::CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);

	{
#if defined(__USE_ADDBACK_DUAL) || defined(__USE_ADDBACK_DISPATCH)
		IAddBackPtr pAddBack(__uuidof(AddBack));
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
		IAddBackPtr pAddBack(__uuidof(CAddBack));
#endif

		CComObject<CEventHandler>* handler;
		CComObject<CEventHandler>::CreateInstance(&handler);

		if (!handler->HookEvent(pAddBack))
		{
			std:: cout << "이벤트를 받을 수 없습니다." << std::endl;
			return 1;
		}

		SHORT sum;
		std::cout << "AddEnd : 20" << std::endl;
		pAddBack->put_AddEnd(20);
		pAddBack->Add();
		pAddBack->get_Sum(&sum);
		std::cout << "Sum: " << sum << std::endl;

		std::cout << "AddEnd: 30" << std::endl;
		pAddBack->put_AddEnd(30);
		pAddBack->Add();
		pAddBack->get_Sum(&sum);
		std::cout << "Sum: " << sum << std::endl;

		handler->UnhookEvent(pAddBack);

		pAddBack = 0;
		handler = 0;
	}

	_Module.Term();
	::CoUninitialize();

	return 0;
}

