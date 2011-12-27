#pragma once

[event_receiver(com)]
class CEventHandler :
	public CComObjectRootEx<CComSingleThreadModel>,
#if defined(__USE_ADDBACK_DUAL)
	public IDispatchImpl<IAddBackEvents, &__uuidof(IAddBackEvents), &__uuidof(__AddBackServerLib)>
#elif defined(__USE_ADDBACK_DISPATCH)
	public IUnknown,
	public IDispEventImpl<1, CEventHandler, &__uuidof(_IAddBackEvents), &__uuidof(__AddBackServerLib), 1, 0>
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	public IUnknown,
	public IDispEventImpl<1, CEventHandler, &__uuidof(_IAddBackEvents), &__uuidof(__AddBack_byAttrib), 1, 0>
#endif
{
public:
	CEventHandler(void);
	~CEventHandler(void);

public:
BEGIN_COM_MAP(CEventHandler)
#if defined(__USE_ADDBACK_DUAL)
	COM_INTERFACE_ENTRY(IDispatch)
	COM_INTERFACE_ENTRY(IAddBackEvents)
#elif defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	COM_INTERFACE_ENTRY(IUnknown)
#endif
END_COM_MAP()

#if defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
BEGIN_SINK_MAP(CEventHandler)
	SINK_ENTRY_EX(1, __uuidof(_IAddBackEvents), 1, onChangedAddEnd)
	SINK_ENTRY_EX(1, __uuidof(_IAddBackEvents), 2, onChangedSum)
END_SINK_MAP()
#endif

public:
	bool HookEvent(IUnknown* pSource);
	bool UnhookEvent(IUnknown* pSource);

#if defined(__USE_ADDBACK_DUAL)
	STDMETHOD(raw_ChangedAddEnd)(short newVal);
	STDMETHOD(raw_ChangedSum)(short newVal);
#elif defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	STDMETHOD(onChangedAddEnd)(short newVal);
	STDMETHOD(onChangedSum)(short newVal);
#endif

private:
#if defined(__USE_ADDBACK_DUAL)
	DWORD cookie_;
#endif
};
