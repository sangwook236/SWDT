// AddFront_byAttribDoc.h : CAddFront_byAttribDoc 클래스의 인터페이스
//


#pragma once


class CAddFront_byAttribDoc : public CDocument
{
protected: // serialization에서만 만들어집니다.
	CAddFront_byAttribDoc();
	DECLARE_DYNCREATE(CAddFront_byAttribDoc)

// 특성입니다.
public:

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);

// 구현입니다.
public:
	virtual ~CAddFront_byAttribDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()

	// 생성된 OLE 디스패치 맵 함수

	DECLARE_DISPATCH_MAP()
	DECLARE_INTERFACE_MAP()
};


