// Microsoft.cpp : 응용 프로그램에 대한 클래스 동작을 정의합니다.
//

#include "stdafx.h"
#include "Microsoft.h"
#include "MainFrm.h"

#include "MicrosoftDoc.h"
#include "MicrosoftView.h"

#include "ConsoleWindow.h"
#include <string>
#include <iostream>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMicrosoftApp

BEGIN_MESSAGE_MAP(CMicrosoftApp, CWinApp)
	ON_COMMAND(ID_APP_ABOUT, &CMicrosoftApp::OnAppAbout)
	// 표준 파일을 기초로 하는 문서 명령입니다.
	ON_COMMAND(ID_FILE_NEW, &CWinApp::OnFileNew)
	ON_COMMAND(ID_FILE_OPEN, &CWinApp::OnFileOpen)
	// 표준 인쇄 설정 명령입니다.
	ON_COMMAND(ID_FILE_PRINT_SETUP, &CWinApp::OnFilePrintSetup)
END_MESSAGE_MAP()


// CMicrosoftApp 생성

CMicrosoftApp::CMicrosoftApp()
{
	// TODO: 여기에 생성 코드를 추가합니다.
	// InitInstance에 모든 중요한 초기화 작업을 배치합니다.
}


// 유일한 CMicrosoftApp 개체입니다.

CMicrosoftApp theApp;


// CMicrosoftApp 초기화

BOOL CMicrosoftApp::InitInstance()
{
	// 응용 프로그램 매니페스트가 ComCtl32.dll 버전 6 이상을 사용하여 비주얼 스타일을
	// 사용하도록 지정하는 경우, Windows XP 상에서 반드시 InitCommonControlsEx()가 필요합니다. 
	// InitCommonControlsEx()를 사용하지 않으면 창을 만들 수 없습니다.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// 응용 프로그램에서 사용할 모든 공용 컨트롤 클래스를 포함하도록
	// 이 항목을 설정하십시오.
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();

	// OLE 라이브러리를 초기화합니다.
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}
	AfxEnableControlContainer();
	// 표준 초기화
	// 이들 기능을 사용하지 않고 최종 실행 파일의 크기를 줄이려면
	// 아래에서 필요 없는 특정 초기화
	// 루틴을 제거해야 합니다.
	// 해당 설정이 저장된 레지스트리 키를 변경하십시오.
	// TODO: 이 문자열을 회사 또는 조직의 이름과 같은
	// 적절한 내용으로 수정해야 합니다.
	SetRegistryKey(_T("로컬 응용 프로그램 마법사에서 생성된 응용 프로그램"));
	LoadStdProfileSettings(4);  // MRU를 포함하여 표준 INI 파일 옵션을 로드합니다.
	// 응용 프로그램의 문서 템플릿을 등록합니다. 문서 템플릿은
	//  문서, 프레임 창 및 뷰 사이의 연결 역할을 합니다.
	CSingleDocTemplate* pDocTemplate;
	pDocTemplate = new CSingleDocTemplate(
		IDR_MAINFRAME,
		RUNTIME_CLASS(CMicrosoftDoc),
		RUNTIME_CLASS(CMainFrame),       // 주 SDI 프레임 창입니다.
		RUNTIME_CLASS(CMicrosoftView));
	if (!pDocTemplate)
		return FALSE;
	AddDocTemplate(pDocTemplate);



	// 표준 셸 명령, DDE, 파일 열기에 대한 명령줄을 구문 분석합니다.
	CCommandLineInfo cmdInfo;
	ParseCommandLine(cmdInfo);


	// 명령줄에 지정된 명령을 디스패치합니다.
	// 응용 프로그램이 /RegServer, /Register, /Unregserver 또는 /Unregister로 시작된 경우 FALSE를 반환합니다.
	if (!ProcessShellCommand(cmdInfo))
		return FALSE;

	// 창 하나만 초기화되었으므로 이를 표시하고 업데이트합니다.
	m_pMainWnd->ShowWindow(SW_SHOW);
	m_pMainWnd->UpdateWindow();
	// 접미사가 있을 경우에만 DragAcceptFiles를 호출합니다.
	//  SDI 응용 프로그램에서는 ProcessShellCommand 후에 이러한 호출이 발생해야 합니다.

	//------------------------------------------------------------------------------------
	// output debug
	OutputDebugString(_T(">>>>> initialize console window\n"));
	//------------------------------------------------------------------------------------
	// initialize console window
	ConsoleWindow::initialize();

	//------------------------------------------------------------------------------------
	// use console window
	if (ConsoleWindow::getInstance().isValid())
	{
		HANDLE stdOutputHandle = GetStdHandle(STD_OUTPUT_HANDLE);
		HANDLE stdInputHandle = GetStdHandle(STD_INPUT_HANDLE);
		if (INVALID_HANDLE_VALUE != stdOutputHandle && INVALID_HANDLE_VALUE != stdInputHandle)
		{
			//
			DWORD dwCharsWritten = 0;
			WriteConsole(stdOutputHandle, _T("enter some text ...\n"), 20, &dwCharsWritten, NULL);

			const DWORD dwCharsToRead = 1024;
			DWORD dwCharsRead = 0;
			TCHAR buf[dwCharsToRead] = { 0, };
			ReadConsole(stdInputHandle, buf, dwCharsToRead, &dwCharsRead, NULL);
			if (dwCharsRead)
				WriteConsole(stdOutputHandle, buf, dwCharsRead, &dwCharsWritten, NULL);
		}

		//
		std::cout << "enter some text ..." << std::endl;
		std::string str;
		std::getline(std::cin, str);
		if (!str.empty())
			std::cout << str << std::endl;
	}

	return TRUE;
}

int CMicrosoftApp::ExitInstance()
{
	//------------------------------------------------------------------------------------
	// output debug
	OutputDebugString(_T(">>>>> finalize console window\n"));
	//------------------------------------------------------------------------------------
	// finalize console window
	ConsoleWindow::finalize();

	return CWinApp::ExitInstance();
}



// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()

// 대화 상자를 실행하기 위한 응용 프로그램 명령입니다.
void CMicrosoftApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}


// CMicrosoftApp 메시지 처리기
