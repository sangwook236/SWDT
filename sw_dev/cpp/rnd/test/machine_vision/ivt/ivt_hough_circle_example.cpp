//include "stdafx.h"
#include <VideoCapture/BitmapCapture.h>
#include <Interfaces/ApplicationHandlerInterface.h>
#include <Interfaces/MainWindowInterface.h>
#include <Interfaces/MainWindowEventInterface.h>
#include <gui/GUIFactory.h>
#include <Image/ByteImage.h>
#include <Image/ImageProcessor.h>
#include <Helpers/helpers.h>
#include <iostream>
#include <string>
#include <cstdio>


namespace {
namespace local {

class CHoughCirclesDemo : public CMainWindowEventInterface
{
public:
	// constructor
	CHoughCirclesDemo() : m_nCannyLowThreshold(50), m_nCannyHighThreshold(200), m_nCirclesToExtract(1),
		m_nMinRadius(105), m_nMaxRadius(115), m_bShowEdges(false)
	{
	}

	// this is called when the value of one of the sliders is changed
	void ValueChanged(WIDGET_HANDLE pWidget, int nValue) 
	{
		if (pWidget == m_pSliderCannyLow)
			m_nCannyLowThreshold = nValue;
		else if (pWidget == m_pSliderCannyHigh)
			m_nCannyHighThreshold = nValue;
		else if (pWidget == m_pSliderMinRadius)
			m_nMinRadius = nValue;
		else if (pWidget == m_pSliderMaxRadius)
			m_nMaxRadius = nValue;
		else if (pWidget == m_pSliderLinesToExtract)
			m_nCirclesToExtract = nValue;
	}

	// this is called when a button is pushed
	void ButtonPushed(WIDGET_HANDLE pWidget)
	{
		if (pWidget == m_pButton)
		{
			m_bShowEdges = !m_bShowEdges;

			if (m_bShowEdges)
				m_pMainWindow->SetText(m_pButton, "Show Result");
			else
				m_pMainWindow->SetText(m_pButton, "Show Edges");
		}
	}

	// init application and run
	int Run(const std::string &input_filename)
	{
		CBitmapCapture capture(input_filename.c_str());

		// open camera
		if (!capture.OpenCamera())
		{
			std::cerr << "error: could not open camera" << std::endl;
			return 1;
		}

		const int width = capture.GetWidth();
		const int height = capture.GetHeight();

		// create temp image for the image processing
		CByteImage image(width, height, capture.GetType());
		CByteImage grayImage(width, height, CByteImage::eGrayScale);
		CByteImage tempImage(width, height, CByteImage::eGrayScale);
		CByteImage visualizationImage(width, height, CByteImage::eRGB24);
		CByteImage *pImage = &image;


		// create an application handler
		CApplicationHandlerInterface *pApplicationHandler = CreateApplicationHandler();
		pApplicationHandler->Reset();

		// create a main window
		m_pMainWindow = CreateMainWindow(0, 0, width, height + 190, "Hough Line Detection Demo");

		// events are sent to this class, hence this class needs to have the CMainWindowEventInterface
		m_pMainWindow->SetEventCallback(this);

		// create an image widget to display a window
		WIDGET_HANDLE pImageWidget = m_pMainWindow->AddImage(0, 190, width, height);

		// add a label and a slider for the low threshold
		WIDGET_HANDLE pLabelCannyLow = m_pMainWindow->AddLabel(15, 15, 200, 30, "Canny low threshold: 0");
		m_pSliderCannyLow = m_pMainWindow->AddSlider(15, 30, 200, 40, 0, 1020, 102, m_nCannyLowThreshold);

		// add a label and a slider for the high threshold
		WIDGET_HANDLE pLabelCannyHigh = m_pMainWindow->AddLabel(15, 70, 200, 30, "Canny high threshold: 0");
		m_pSliderCannyHigh = m_pMainWindow->AddSlider(15, 85, 200, 40, 0, 1020, 102, m_nCannyHighThreshold);

		// add a label and a slider for the number of lines to extract
		WIDGET_HANDLE pLabelLines = m_pMainWindow->AddLabel(260, 15, 200, 30, "Circles to extract: 0 lines");
		m_pSliderLinesToExtract = m_pMainWindow->AddSlider(260, 30, 200, 40, 0, 30, 5, m_nCirclesToExtract);

		// add labels/sliders for specifying the radius interval of interest
		WIDGET_HANDLE pLabelMinRadius = m_pMainWindow->AddLabel(260, 70, 200, 30, "Min radius: 0");
		m_pSliderMinRadius = m_pMainWindow->AddSlider(260, 85, 200, 40, 1, 200, 5, m_nMinRadius);
		WIDGET_HANDLE pLabelMaxRadius = m_pMainWindow->AddLabel(260, 125, 200, 30, "Max radius: 0");
		m_pSliderMaxRadius = m_pMainWindow->AddSlider(260, 140, 200, 40, 1, 200, 5, m_nMaxRadius);

		// add a button to toggle between the original image and the processed one
		m_pButton = m_pMainWindow->AddButton(510, 80, 110, 35, "Show Edges");

		// add a labels to display processing stats
		WIDGET_HANDLE pLabelMS = m_pMainWindow->AddLabel(560, 15, 70, 20, "0 ms");
		WIDGET_HANDLE pLabelFPS = m_pMainWindow->AddLabel(560, 45, 70, 20, "0 fps");

		// make the window visible
		m_pMainWindow->Show();


		char buffer[1024];

		CVec3dArray resultListCircles(50);
		CDynamicArrayTemplate<int> resultHits(50);
		CVec2dArray edgePoints(10000), edgeDirections(10000);

		// main loop
		while (!pApplicationHandler->ProcessEventsAndGetExit())
		{
			if (!capture.CaptureImage(&pImage))
				break;

			// this is for visualization purposes only
			ImageProcessor::ConvertImage(pImage, &visualizationImage);

			get_timer_value(true);

			// convert input image to grayscale image
			ImageProcessor::ConvertImage(&image, &tempImage, true);

			// smooth image
			ImageProcessor::GaussianSmooth3x3(&tempImage, &grayImage);

			// detect edges with Canny edge detector
			ImageProcessor::Canny(&grayImage, edgePoints, edgeDirections, m_nCannyLowThreshold, m_nCannyHighThreshold);

			// detect lines with Hough transform
			ImageProcessor::HoughTransformCircles(edgePoints, edgeDirections, width, height, m_nMinRadius, m_nMaxRadius, m_nCirclesToExtract, 1, resultListCircles, resultHits, &visualizationImage);

			const unsigned int t = get_timer_value();

			// display the speed stats
			sprintf(buffer, "%2.2f ms", t / 1000.0f);
			m_pMainWindow->SetText(pLabelMS, buffer);
			sprintf(buffer, "%3.2f fps", 1000000.0f / t);
			m_pMainWindow->SetText(pLabelFPS, buffer);
			sprintf(buffer, "Canny low threshold: %i", m_nCannyLowThreshold);
			m_pMainWindow->SetText(pLabelCannyLow, buffer);
			sprintf(buffer, "Canny high threshold: %i", m_nCannyHighThreshold);
			m_pMainWindow->SetText(pLabelCannyHigh, buffer);
			sprintf(buffer, "Min radius: %i", m_nMinRadius);
			m_pMainWindow->SetText(pLabelMinRadius, buffer);
			sprintf(buffer, "Max radius: %i", m_nMaxRadius);
			m_pMainWindow->SetText(pLabelMaxRadius, buffer);
			sprintf(buffer, "Circles to extract: %i", m_nCirclesToExtract);
			m_pMainWindow->SetText(pLabelLines, buffer);

			// display either the original image or the processed image
			if (m_bShowEdges)
			{
				ImageProcessor::Canny(&grayImage, &grayImage, m_nCannyLowThreshold, m_nCannyHighThreshold);
				m_pMainWindow->SetImage(pImageWidget, &grayImage);
			}
			else
				m_pMainWindow->SetImage(pImageWidget, &visualizationImage);
		}

		delete m_pMainWindow;
		delete pApplicationHandler;

		return 0;
	}


private:
	// private attributes
	CMainWindowInterface *m_pMainWindow;
	WIDGET_HANDLE m_pSliderCannyLow, m_pSliderCannyHigh;
	WIDGET_HANDLE m_pSliderMinRadius, m_pSliderMaxRadius;
	WIDGET_HANDLE m_pSliderLinesToExtract;
	WIDGET_HANDLE m_pButton;

	bool m_bShowEdges;
	int m_nCannyLowThreshold, m_nCannyHighThreshold;
	int m_nMinRadius, m_nMaxRadius;
	int m_nCirclesToExtract;
};

}  // namespace local
}  // unnamed namespace

namespace my_ivt {

// [ref] ${IVT_HOME}/examples/HoughCirclesDemo/main.cpp
void hough_circle_example()
{
	const std::string input_filename("./machine_vision_data/ivt/geometric_scene.bmp");

	local::CHoughCirclesDemo demo;
	const int retval = demo.Run(input_filename);
}

}  // namespace my_ivt
