//include "stdafx.h"
#if defined WIN32
#include <VideoCapture/VFWCapture.h>
#elif defined __APPLE__
#include <VideoCapture/QuicktimeCapture.h>
#else
#include <VideoCapture/Linux1394Capture2.h>
#endif
#include <Interfaces/ApplicationHandlerInterface.h>
#include <Interfaces/MainWindowInterface.h>
#include <Interfaces/MainWindowEventInterface.h>
#include <gui/GUIFactory.h>
#include <Tracking/KLTTracker.h>
#include <Helpers/helpers.h>
#include <Image/PrimitivesDrawer.h>
#include <Image/ImageProcessor.h>
#include <Math/Math2d.h>
#include <iostream>
#include <cstdio>


namespace {
namespace local {

class CKLTTrackerDemo : public CMainWindowEventInterface
{
public:
	// constructor
	CKLTTrackerDemo() : m_fQualityThreshold(0.005f), m_nMaxPoints(500), m_nPoints(0)
	{
	}

	// this is called when the value of one of the sliders is changed
	void ValueChanged(WIDGET_HANDLE pWidget, int nValue)
	{
		if (pWidget == m_pSlider1)
			m_fQualityThreshold = nValue / 10000.0f;
		else if (pWidget == m_pSlider2)
			m_nMaxPoints = nValue;
	}

	// this is called when a button is pressed
	void ButtonPushed(WIDGET_HANDLE widget)
	{
		if (widget == m_pButtonReInit)
			m_nPoints = 0;
	}

	// init application and run
	int Run()
	{
		// create capture object
#if defined WIN32
		CVFWCapture capture(0);
#elif defined __APPLE__
		CQuicktimeCapture capture(CVideoCaptureInterface::e640x480);
#else
		CLinux1394Capture2 capture(-1, CVideoCaptureInterface::e640x480, CVideoCaptureInterface::eRGB24)
#endif

			// open camera
			if (!capture.OpenCamera())
			{
				printf("error: could not open camera\n");
				printf("press return to quit\n");
				char szTemp[1024];
				scanf("%c", szTemp);
				return false;
			}

			const int width = capture.GetWidth();
			const int height = capture.GetHeight();

			CByteImage image(width, height, capture.GetType());
			CByteImage grayImage(width, height, CByteImage::eGrayScale);
			CByteImage *pImage = &image;

			// create an application handler
			CApplicationHandlerInterface *pApplicationHandler = CreateApplicationHandler();
			pApplicationHandler->Reset();

			// create a main window
			CMainWindowInterface *pMainWindow = CreateMainWindow(0, 0, width, height + 100, "KLT Tracker Demo");

			// events are sent to this class, hence this class needs to have the CMainWindowEventInterface
			pMainWindow->SetEventCallback(this);

			// create an image widget to display a window
			WIDGET_HANDLE pImageWidget = pMainWindow->AddImage(0, 100, width, height);

			// add a label and a slider for the quality
			WIDGET_HANDLE pLabel1 = pMainWindow->AddLabel(10, 10, 150, 30, "Quality");
			m_pSlider1 = pMainWindow->AddSlider(10, 50, 150, 40, 0, 100, 10, int(m_fQualityThreshold * 10000.0f + 0.5f));

			// add a label and a slider for the number of interest points
			WIDGET_HANDLE pLabel2 = pMainWindow->AddLabel(200, 10, 200, 30, "Number of points");
			m_pSlider2 = pMainWindow->AddSlider(200, 30, 150, 40, 0, 1000, 50, m_nMaxPoints);

			// add a button for re-initializing the features to be tracker
			m_pButtonReInit = pMainWindow->AddButton(400, 50, 100, 40, "Re-Init");

			// add a labels to display processing stats
			WIDGET_HANDLE pLabel3 = pMainWindow->AddLabel(520, 10, 120, 20, "666 ms");
			WIDGET_HANDLE pLabel4 = pMainWindow->AddLabel(520, 40, 120, 20, "666 fps");
			WIDGET_HANDLE pLabel5 = pMainWindow->AddLabel(520, 70, 120, 20, "666 points");

			// make the window visible
			pMainWindow->Show();

			CKLTTracker tracker(width, height, 3, 10);
			Vec2d points[1000];

			char buffer[1024];

			while (!pApplicationHandler->ProcessEventsAndGetExit())
			{
				if (!capture.CaptureImage(&pImage))
					break;

				ImageProcessor::ConvertImage(pImage, &grayImage, true);

				if (m_nPoints == 0)
					m_nPoints = ImageProcessor::CalculateHarrisInterestPoints(&grayImage, points, m_nMaxPoints, m_fQualityThreshold, 10.0f);

				get_timer_value(true);
				tracker.Track(&grayImage, points, m_nPoints, points);
				const unsigned int t = get_timer_value();

				for (int i = 0; i < m_nPoints; i++)
				{
					if (points[i].x > 0.0f && points[i].y > 0.0f) // points for that track has got lost are marked with x = y = -1.0f
						PrimitivesDrawer::DrawCircle(pImage, points[i], 3, 0, 255, 0, -1);
				}

				pMainWindow->SetImage(pImageWidget, pImage);

				// display some information
				sprintf(buffer, "Quality = %.4f", m_fQualityThreshold);
				pMainWindow->SetText(pLabel1, buffer);
				sprintf(buffer, "Max number of points = %d", m_nMaxPoints);
				pMainWindow->SetText(pLabel2, buffer);

				// display the speed stats
				sprintf(buffer, "%.2f ms", t / 1000.0f);
				pMainWindow->SetText(pLabel3, buffer);
				sprintf(buffer, "%.2f fps", 1000000.0f / t);
				pMainWindow->SetText(pLabel4, buffer);
				sprintf(buffer, "%d points", m_nPoints);
				pMainWindow->SetText(pLabel5, buffer);
			}

			delete pMainWindow;
			delete pApplicationHandler;

			return 0;
	}


private:
	// private attributes
	float m_fQualityThreshold;
	int m_nMaxPoints;
	int m_nPoints;
	WIDGET_HANDLE m_pSlider1, m_pSlider2;
	WIDGET_HANDLE m_pButtonReInit;
};

}  // namespace local
}  // unnamed namespace

namespace my_ivt {

// [ref] ${IVT_HOME}/examples/KLTTrackerDemo/main.cpp
void klt_tracker_example()
{
	local::CKLTTrackerDemo demo;
	const int retval = demo.Run();
}

}  // namespace my_ivt
