// AddBack.h : CAddBack의 선언입니다.

#pragma once
#include "resource.h"       // 주 기호입니다.



#if defined(_WIN32_WCE) && !defined(_CE_DCOM) && !defined(_CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA)
#error "단일 스레드 COM 개체는 전체 DCOM 지원을 포함하지 않는 Windows Mobile 플랫폼과 같은 Windows CE 플랫폼에서 제대로 지원되지 않습니다. ATL이 단일 스레드 COM 개체의 생성을 지원하고 단일 스레드 COM 개체 구현을 사용할 수 있도록 _CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA를 정의하십시오. rgs 파일의 스레딩 모델은 DCOM Windows CE가 아닌 플랫폼에서 지원되는 유일한 스레딩 모델이므로 'Free'로 설정되어 있습니다."
#endif


// IAddBack
[
	object,
	uuid("493B7227-97B4-426D-9F47-B121E0DA6D3B"),
	dual,	helpstring("IAddBack Interface"),
	pointer_default(unique)
]
__interface IAddBack : IDispatch
{
	[propget, id(1), helpstring("속성 AddEnd")] HRESULT AddEnd([out, retval] SHORT* pVal);
	[propput, id(1), helpstring("속성 AddEnd")] HRESULT AddEnd([in] SHORT newVal);
	[propget, id(2), helpstring("속성 Sum")] HRESULT Sum([out, retval] SHORT* pVal);
	[id(3), helpstring("메서드 Add")] HRESULT Add(void);
	[id(4), helpstring("메서드 AddTen")] HRESULT AddTen(void);
	[id(5), helpstring("메서드 Clear")] HRESULT Clear(void);
};


// _IAddBackEvents
[
	dispinterface,
	uuid("CB43E856-D48D-4688-92EC-060F5C2EDDCC"),
	helpstring("_IAddBackEvents Interface")
]
__interface _IAddBackEvents
{
	[id(1), helpstring("ChangedAddEnd 이벤트")] HRESULT ChangedAddEnd([in] SHORT newVal);
	[id(2), helpstring("ChangedSum 이벤트")] HRESULT ChangedSum([in] SHORT newVal);
};


// CAddBack

[
	coclass,
	default(IAddBack, _IAddBackEvents),
	threading("apartment"),
	event_source("com"),
	vi_progid("AddBack_byAttrib.AddBack"),
	progid("AddBack_byAttrib.AddBack.1"),
	version(1.0),
	uuid("A6F1D48F-0B55-43CF-8390-516AB184160D"),
	helpstring("AddBack Class")
]
class ATL_NO_VTABLE CAddBack :
	public IAddBack
{
public:
	CAddBack()
	{
		sum_ = 0;
		addEnd_ = 10;
	}

	__event __interface _IAddBackEvents;


	DECLARE_PROTECT_FINAL_CONSTRUCT()

	HRESULT FinalConstruct()
	{
		return S_OK;
	}

	void FinalRelease()
	{
	}

public:

	STDMETHOD(get_AddEnd)(SHORT* pVal);
	STDMETHOD(put_AddEnd)(SHORT newVal);
	STDMETHOD(get_Sum)(SHORT* pVal);
	STDMETHOD(Add)(void);
	STDMETHOD(AddTen)(void);
	STDMETHOD(Clear)(void);

private:
	SHORT sum_;
	SHORT addEnd_;
};

