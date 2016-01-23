/***************************************************************
 * Name:      wxwidgetsMain.cpp
 * Purpose:   Code for Application Frame
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

#include "wxwidgetsMain.h"
#include <wx/msgdlg.h>


wxwidgetsFrame::wxwidgetsFrame(wxFrame *frame)
    : GUIFrame(frame)
{
#if wxUSE_STATUSBAR
    statusBar->SetStatusText(_("Message #0 in Statusbar"), 0);
    statusBar->SetStatusText(_("Message #1 in Statusbar"), 1);
#endif
}

wxwidgetsFrame::~wxwidgetsFrame()
{
}

void wxwidgetsFrame::OnClose(wxCloseEvent &event)
{
    Destroy();
}

void wxwidgetsFrame::OnQuit(wxCommandEvent &event)
{
    Destroy();
}

void wxwidgetsFrame::OnAbout(wxCommandEvent &event)
{
    wxMessageBox(_("Message in MsgBox"), _("Welcome to..."), wxICON_EXCLAMATION);
}

void wxwidgetsFrame::OnCancelButtonClick(wxCommandEvent& event)
{
    wxMessageBox(_("Cancel is pressed"), _("Information"), wxICON_INFORMATION);
}

void wxwidgetsFrame::OnOKButtonClick(wxCommandEvent& event)
{
    wxMessageBox(_("OK is pressed"), _("Information"), wxICON_INFORMATION);
}
