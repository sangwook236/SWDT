// AddBack.cpp : CAddBack의 구현입니다.

#include "stdafx.h"
#include "AddBack.h"


// CAddBack


STDMETHODIMP CAddBack::get_AddEnd(SHORT* pVal)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	*pVal = addEnd_;

	return S_OK;
}

STDMETHODIMP CAddBack::put_AddEnd(SHORT newVal)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	addEnd_ = newVal;
	FireChangedAddEnd();

	return S_OK;
}

STDMETHODIMP CAddBack::get_Sum(SHORT* pVal)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	*pVal = sum_;

	return S_OK;
}

STDMETHODIMP CAddBack::Add(void)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	sum_ += addEnd_;
	FireChangedSum();

	return S_OK;
}

STDMETHODIMP CAddBack::AddTen(void)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	sum_ += 10;
	FireChangedSum();

	return S_OK;
}

STDMETHODIMP CAddBack::Clear(void)
{
	// TODO: 여기에 구현 코드를 추가합니다.
	sum_ = 0;
	FireChangedSum();

	return S_OK;
}

void CAddBack::FireChangedAddEnd()
{
	CProxy_IAddBackEvents<CAddBack>::Fire_ChangedAddEnd(addEnd_);
	CProxyIAddBackEvents<CAddBack>::Fire_ChangedAddEnd(addEnd_);
}

void CAddBack::FireChangedSum()
{
	CProxy_IAddBackEvents<CAddBack>::Fire_ChangedSum(sum_);
	CProxyIAddBackEvents<CAddBack>::Fire_ChangedSum(sum_);
}
