========================================================================
    ACTIVE TEMPLATE LIBRARY : AddBackServer 프로젝트 개요
========================================================================

사용자가 DLL(동적 연결 라이브러리)을 만드는 데 있어 출발점으로 사용하도록
응용 프로그램 마법사에서 이 AddBackServer 프로젝트를 만들었습니다.


이 파일에는 프로젝트를 구성하는 각각의 파일에 들어 있는 요약 설명이 포함되어 있습니다.

AddBackServer.vcproj
    응용 프로그램 마법사를 사용하여 생성한 VC++ 프로젝트의 기본 프로젝트 파일입니다. 
    해당 파일을 생성한 Visual C++의 버전 정보를 비롯하여
    응용 프로그램 마법사에서 선택한 플랫폼, 구성 및 프로젝트 기능에 대한 정보가 들어 있습니다.

AddBackServer.idl
    이 파일에는 형식 라이브러리의 IDL 정의 및,
    프로젝트에서 정의한 인터페이스와 보조 클래스가 들어 있습니다.
    이 파일은 MIDL 컴파일러에 의해 처리되어 다음을 생성합니다.
        C++ 인터페이스 정의 및 GUID 선언 (AddBackServer.h)
        GUID 정의                    (AddBackServer_i.c)
        형식 라이브러리                (AddBackServer.tlb)
        마샬링 코드                            (AddBackServer_p.c 및 dlldata.c)

AddBackServer.h
    이 파일에는 AddBackServer.idl에서 정의된 항목의 C++ 인터페이스가 정의되어 있고
    GUID가 선언되어 있으며, 컴파일하는 동안 MIDL에 의해 다시 생성됩니다.

AddBackServer.cpp
    이 파일에는 오브젝트 맵이 들어 있으며 DLL 내보내기가 구현되어 있습니다.

AddBackServer.rc
    프로그램에서 사용하는 모든 Microsoft Windows 리소스의 목록입니다.

AddBackServer.def
    이 모듈 정의 파일에서는 DLL에서 필요한 내보내기에 대한 정보를 링커에 제공합니다.
    다음에 대한 내보내기가 포함되어 있습니다.
        DllGetClassObject  
        DllCanUnloadNow    
        GetProxyDllInfo    
        DllRegisterServer	
        DllUnregisterServer

/////////////////////////////////////////////////////////////////////////////
기타 표준 파일:

StdAfx.h 및 StdAfx.cpp는
    AddBackServer.pch라는 이름의 PCH(미리 컴파일된 헤더) 파일과
    StdAfx.obj라는 이름의 미리 컴파일된 형식 파일을 빌드하는 데 사용됩니다.

Resource.h
    리소스 ID를 정의하는 표준 헤더 파일입니다.

/////////////////////////////////////////////////////////////////////////////
프록시/스텁 DLL 프로젝트 및 모듈 정의 파일입니다.

AddBackServerps.vcproj
    필요한 경우 프록시/스텁 DLL을 빌드할 수 있는 프로젝트 파일입니다.
	기본 프로젝트의 IDL 파일에는 인터페이스가 하나 이상 들어 있어야 하며
	프록시/스텁 DLL을 빌드하기 전에 해당 IDL 파일을 먼저 컴파일해야 합니다. 이렇게 하면
	프록시/스텁 DLL을 빌드하는 데 필요한 dlldata.c, AddBackServer_i.c 및
	AddBackServer_p.c가 생성됩니다.

AddBackServerps.def
    이 모듈 정의 파일에서는 프록시/스텁에서 필요한 내보내기에 대한 정보를 링커에
    제공합니다.

/////////////////////////////////////////////////////////////////////////////
