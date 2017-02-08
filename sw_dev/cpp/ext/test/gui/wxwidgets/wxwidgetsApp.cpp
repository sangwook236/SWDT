#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include "wxwidgetsApp.h"
#include "wxwidgetsMain.h"

IMPLEMENT_APP(wxwidgetsApp);

bool wxwidgetsApp::OnInit()
{
    wxwidgetsFrame* frame = new wxwidgetsFrame(0L);
    
    frame->Show();
    
    return true;
}
