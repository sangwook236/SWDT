/***************************************************************
 * Name:      wxwidgetsApp.cpp
 * Purpose:   Code for Application Class
 * Author:     ()
 * Created:   2016-01-17
 * Copyright:  ()
 * License:
 **************************************************************/

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "wxwidgetsApp.h"
#include "wxwidgetsMain.h"

IMPLEMENT_APP(wxwidgetsApp);

bool wxwidgetsApp::OnInit()
{
    wxwidgetsFrame* frame = new wxwidgetsFrame(0L);
    
    frame->Show();
    
    return true;
}
