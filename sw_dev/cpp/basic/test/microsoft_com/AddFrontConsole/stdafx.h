// stdafx.h : 자주 사용하지만 자주 변경되지는 않는
// 표준 시스템 포함 파일 및 프로젝트 관련 포함 파일이
// 들어 있는 포함 파일입니다.
//

#pragma once

#ifndef _WIN32_WINNT		// Windows XP 이상에서만 기능을 사용할 수 있습니다.                   
#define _WIN32_WINNT 0x0501	// 다른 버전의 Windows에 맞도록 적합한 값으로 변경해 주십시오.
#endif						

#include <stdio.h>
#include <tchar.h>

#define __USE_ADDBACK_DUAL 1
//#define __USE_ADDBACK_DISPATCH 1
//#define __USE_ADDBACK_BYATTRIB_DISPATCH 1

//#define _ATL_ATTRIBUTES 1
#include <afxdisp.h>
#if defined(__USE_ADDBACK_DUAL) || defined(__USE_ADDBACK_DISPATCH)
#import "progid:AddBackServer.AddBack.1" no_namespace named_guids
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
#import "progid:AddBack_byAttrib.AddBack.1" no_namespace named_guids
#endif
#include <atlbase.h>
extern CComModule _Module;
#include <atlcom.h>



// TODO: 프로그램에 필요한 추가 헤더는 여기에서 참조합니다.
