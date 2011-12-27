// AddBack.h : CAddBack의 선언입니다.

#pragma once
#include "resource.h"       // 주 기호입니다.

#include "AddBackServer.h"
#include "_IAddBackEvents_CP.h"
#include "IAddBackEvents_CP.H"


#if defined(_WIN32_WCE) && !defined(_CE_DCOM) && !defined(_CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA)
#error "단일 스레드 COM 개체는 전체 DCOM 지원을 포함하지 않는 Windows Mobile 플랫폼과 같은 Windows CE 플랫폼에서 제대로 지원되지 않습니다. ATL이 단일 스레드 COM 개체의 생성을 지원하고 단일 스레드 COM 개체 구현을 사용할 수 있도록 _CE_ALLOW_SINGLE_THREADED_OBJECTS_IN_MTA를 정의하십시오. rgs 파일의 스레딩 모델은 DCOM Windows CE가 아닌 플랫폼에서 지원되는 유일한 스레딩 모델이므로 'Free'로 설정되어 있습니다."
#endif



// CAddBack

class ATL_NO_VTABLE CAddBack :
	public CComObjectRootEx<CComSingleThreadModel>,
	public CComCoClass<CAddBack, &CLSID_AddBack>,
	public IConnectionPointContainerImpl<CAddBack>,
	public CProxy_IAddBackEvents<CAddBack>,
	public IDispatchImpl<IAddBack, &IID_IAddBack, &LIBID_AddBackServerLib, /*wMajor =*/ 1, /*wMinor =*/ 0>,
	public CProxyIAddBackEvents<CAddBack>
{
public:
	CAddBack()
	{
		sum_ = 0;
		addEnd_ = 10;
	}

DECLARE_REGISTRY_RESOURCEID(IDR_ADDBACK)


BEGIN_COM_MAP(CAddBack)
	COM_INTERFACE_ENTRY(IAddBack)
	COM_INTERFACE_ENTRY(IDispatch)
	COM_INTERFACE_ENTRY(IConnectionPointContainer)
END_COM_MAP()

BEGIN_CONNECTION_POINT_MAP(CAddBack)
	CONNECTION_POINT_ENTRY(__uuidof(IAddBackEvents))
	CONNECTION_POINT_ENTRY(__uuidof(_IAddBackEvents))
END_CONNECTION_POINT_MAP()


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
	void FireChangedAddEnd();
	void FireChangedSum();

private:
	SHORT sum_;
	SHORT addEnd_;
};

OBJECT_ENTRY_AUTO(__uuidof(AddBack), CAddBack)
