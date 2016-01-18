/***************************************************************
 * Name:      wxwidgetsMain.h
 * Purpose:   Defines Application Frame
 * Author:     ()
 * Created:   2016-01-17
 * Copyright:  ()
 * License:
 **************************************************************/

#ifndef WXWIDGETSMAIN_H
#define WXWIDGETSMAIN_H


#include "GUIFrame.h"

class wxwidgetsFrame: public GUIFrame
{
    public:
        wxwidgetsFrame(wxFrame *frame);
        ~wxwidgetsFrame();
    private:
        virtual void OnClose(wxCloseEvent& event);
        virtual void OnQuit(wxCommandEvent& event);
        virtual void OnAbout(wxCommandEvent& event);
};

#endif // WXWIDGETSMAIN_H
