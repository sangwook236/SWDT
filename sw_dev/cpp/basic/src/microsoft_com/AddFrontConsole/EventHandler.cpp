#include "StdAfx.h"
#include "EventHandler.h"

CEventHandler::CEventHandler(void)
{
}

CEventHandler::~CEventHandler(void)
{
}

bool CEventHandler::HookEvent(IUnknown* pSource)
{
#if defined(__USE_ADDBACK_DUAL)
	const HRESULT hr = AtlAdvise(pSource, GetUnknown(), __uuidof(IAddBackEvents), &cookie_);
	if (FAILED(hr)) return false;
#elif defined(__USE_ADDBACK_DISPATCH)
	DispEventAdvise(pSource, &__uuidof(_IAddBackEvents));
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	__hook(&_IAddBackEvents::ChangedAddEnd, pSource, &CEventHandler::onChangedAddEnd);
	__hook(&_IAddBackEvents::ChangedSum, pSource, &CEventHandler::onChangedSum);
#endif

	return true;
}

bool CEventHandler::UnhookEvent(IUnknown* pSource)
{
#if defined(__USE_ADDBACK_DUAL)
	AtlUnadvise(pSource, __uuidof(IAddBackEvents), cookie_);
#elif defined(__USE_ADDBACK_DISPATCH)
	DispEventUnadvise(pSource, &__uuidof(_IAddBackEvents));
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	__unhook(&_IAddBackEvents::ChangedAddEnd, pSource, &CEventHandler::onChangedAddEnd);
	__unhook(&_IAddBackEvents::ChangedSum, pSource, &CEventHandler::onChangedSum);
#endif

	return true;
}

#if defined(__USE_ADDBACK_DUAL)
STDMETHODIMP CEventHandler::raw_ChangedAddEnd(short newVal)
#elif defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
STDMETHODIMP CEventHandler::onChangedAddEnd(short newVal)
#endif
{
	CString szMsg(_T("AddEnd 속성이 변경되었습니다.\n"));
	CString szTemp;
	szTemp.Format(_T("AddEnd 속성값: %d"), newVal);
	szMsg += szTemp;
	MessageBox(NULL, szMsg.GetBuffer(szMsg.GetLength()), _T("AddEnd 속성값 변경"), MB_OK);
	return S_OK;
}

#if defined(__USE_ADDBACK_DUAL)
STDMETHODIMP CEventHandler::raw_ChangedSum(short newVal)
#elif defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
STDMETHODIMP CEventHandler::onChangedSum(short newVal)
#endif
{
	CString szMsg(_T("Sum 속성이 변경되었습니다.\n"));
	CString szTemp;
	szTemp.Format(_T("Sum 속성값: %d"), newVal);
	szMsg += szTemp;
	MessageBox(NULL, szMsg.GetBuffer(szMsg.GetLength()), _T("Sum 속성값 변경"), MB_OK);
	return S_OK;
}
