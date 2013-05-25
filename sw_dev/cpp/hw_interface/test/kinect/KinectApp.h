#pragma once

#include "DrawDevice.h"
#include "resource.h"
#include <NuiApi.h>

// Direct2D Header Files
#include <d2d1.h>
#include <d2d1helper.h>

// DirectWrite
#include <dwrite.h>

//--S [] 2012/06/09: Sang-Wook Lee
#include <opencv2/opencv.hpp>
#include <boost/smart_ptr.hpp>
#include <fstream>
//--E [] 2012/06/09


#define SZ_APPDLG_WINDOW_CLASS          _T("KinectAppDlgWndClass")
#define WM_USER_UPDATE_FPS              WM_USER
#define WM_USER_UPDATE_COMBO            WM_USER+1
#define WM_USER_UPDATE_TRACKING_COMBO   WM_USER+2

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}

class CKinectApp
{
public:
    /// <summary>
    /// Constructor
    /// </summary>
    CKinectApp();
    /// <summary>
    /// Destructor
    /// </summary>
    ~CKinectApp();

    /// <summary>
    /// Initialize Kinect
    /// </summary>
    /// <returns>S_OK if successful, otherwise an error code</returns>
    HRESULT Nui_Init();
    /// <summary>
    /// Initialize Kinect by instance name
    /// </summary>
    /// <param name="instanceName">instance name of Kinect to initialize</param>
    /// <returns>S_OK if successful, otherwise an error code</returns>
    HRESULT Nui_Init(OLECHAR *instanceName);
    /// <summary>
    /// Uninitialize Kinect
    /// </summary>
    void Nui_UnInit();
    /// <summary>
    /// Zero out member variables
    /// </summary>
    void Nui_Zero();
    /// <summary>
    /// Handle new depth data
    /// </summary>
    /// <returns>true if a frame was processed, false otherwise</returns>
    bool Nui_GotDepthAlert();
    /// <summary>
    /// Handle new RGBA data
    /// </summary>
    /// <returns>true if a frame was processed, false otherwise</returns>
    bool Nui_GotRGBAAlert();
	//--S [] 2013/05/16: Sang-Wook Lee
    /// <summary>
    /// Handle new IR data
    /// </summary>
    /// <returns>true if a frame was processed, false otherwise</returns>
    bool Nui_GotIRAlert();
	//--E [] 2013/05/16: Sang-Wook Lee
    /// <summary>
    /// Handle new skeleton data
    /// </summary>
    bool Nui_GotSkeletonAlert();

    /// <summary>
    /// Blank the skeleton display
    /// </summary>
    void Nui_BlankSkeletonScreen();
    /// <summary>
    /// Draws a skeleton
    /// </summary>
    /// <param name="skel">skeleton to draw</param>
    /// <param name="windowWidth">width (in pixels) of output buffer</param>
    /// <param name="windowHeight">height (in pixels) of output buffer</param>
    void Nui_DrawSkeleton(const NUI_SKELETON_DATA &skel, int windowWidth, int windowHeight);
    /// <summary>
    /// Draws a line between two bones
    /// </summary>
    /// <param name="skel">skeleton to draw bones from</param>
    /// <param name="bone0">bone to start drawing from</param>
    /// <param name="bone1">bone to end drawing at</param>
    void Nui_DrawBone( const NUI_SKELETON_DATA &skel, NUI_SKELETON_POSITION_INDEX bone0, NUI_SKELETON_POSITION_INDEX bone1);

    /// <summary>
    /// Handles window messages, passes most to the class instance to handle
    /// </summary>
    /// <param name="hWnd">window message is for</param>
    /// <param name="uMsg">message</param>
    /// <param name="wParam">message data</param>
    /// <param name="lParam">additional message data</param>
    /// <returns>result of message processing</returns>
    static LRESULT CALLBACK MessageRouter(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    /// <summary>
    /// Handle windows messages for the class instance
    /// </summary>
    /// <param name="hWnd">window message is for</param>
    /// <param name="uMsg">message</param>
    /// <param name="wParam">message data</param>
    /// <param name="lParam">additional message data</param>
    /// <returns>result of message processing</returns>
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    /// <summary>
    /// Callback to handle Kinect status changes, redirects to the class callback handler
    /// </summary>
    /// <param name="hrStatus">current status</param>
    /// <param name="instanceName">instance name of Kinect the status change is for</param>
    /// <param name="uniqueDeviceName">unique device name of Kinect the status change is for</param>
    /// <param name="pUserData">additional data</param>
    static void CALLBACK Nui_StatusProcThunk(HRESULT hrStatus, const OLECHAR *instanceName, const OLECHAR *uniqueDeviceName, void *pUserData);
    /// <summary>
    /// Callback to handle Kinect status changes
    /// </summary>
    /// <param name="hrStatus">current status</param>
    /// <param name="instanceName">instance name of Kinect the status change is for</param>
    /// <param name="uniqueDeviceName">unique device name of Kinect the status change is for</param>
    void CALLBACK Nui_StatusProc(HRESULT hrStatus, const OLECHAR *instanceName, const OLECHAR* uniqueDeviceName);

    HWND m_hWnd;
    HINSTANCE m_hInstance;

    int MessageBoxResource(UINT nID, UINT nType);

private:
    /// <summary>
    /// Updates the combo box that lists Kinects available
    /// </summary>
    void UpdateKinectComboBox();
    /// <summary>
    /// Clears the combo box for selecting active Kinect
    /// </summary>
    void ClearKinectComboBox();

    /// <summary>
    /// Invoked when the user changes the tracking mode
    /// </summary>
    /// <param name="mode">tracking mode to switch to</param>
    void UpdateTrackingMode(int mode);
    /// <summary>
    /// Invoked when the user changes the range
    /// </summary>
    /// <param name="mode">range to switch to</param>
    void UpdateRange(int mode);

    /// <summary>
    /// Invoked when the user changes the selection of tracked skeletons
    /// </summary>
    /// <param name="mode">skelton tracking mode to switch to</param>
    void UpdateTrackedSkeletonSelection(int mode);
    /// <summary>
    /// Sets or clears the specified skeleton tracking flag
    /// </summary>
    /// <param name="flag">flag to set or clear</param>
    /// <param name="value">true to set, false to clear</param>
    void UpdateSkeletonTrackingFlag(DWORD flag, bool value);
    /// <summary>
    /// Sets or clears the specified depth stream flag
    /// </summary>
    /// <param name="flag">flag to set or clear</param>
    /// <param name="value">true to set, false to clear</param>
    void UpdateDepthStreamFlag(DWORD flag, bool value);
    /// <summary>
    /// Determines which skeletons to track and tracks them
    /// </summary>
    /// <param name="skel">skeleton frame information</param>
    void UpdateTrackedSkeletons(const NUI_SKELETON_FRAME &skel);

    /// <summary>
    /// Ensure necessary Direct2d resources are created
    /// </summary>
    /// <returns>S_OK if successful, otherwise an error code</returns>
    HRESULT EnsureDirect2DResources();
    /// <summary>
    /// Dispose Direct2d resources 
    /// </summary>
    void DiscardDirect2DResources();

    /// <summary>
    /// Converts a skeleton point to screen space
    /// </summary>
    /// <param name="skeletonPoint">skeleton point to tranform</param>
    /// <param name="width">width (in pixels) of output buffer</param>
    /// <param name="height">height (in pixels) of output buffer</param>
    /// <returns>point in screen-space</returns>
    D2D1_POINT_2F SkeletonToScreen(Vector4 skeletonPoint, int width, int height);

	//--S [] 2012/06/09: Sang-Wook Lee
	//static LPOFNHOOKPROC HookProcCenterDialog(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam);
	LRESULT CALLBACK OnFileOpen(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	LRESULT CALLBACK OnFileSave(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	//--E [] 2012/06/09

    /// <summary>
    /// Thread to handle Kinect processing, calls class instance thread processor
    /// </summary>
    /// <param name="pParam">instance pointer</param>
    /// <returns>always 0</returns>
	static DWORD WINAPI Nui_ProcessThread(LPVOID pParam);
    /// <summary>
    /// Thread to handle Kinect processing
    /// </summary>
    /// <returns>always 0</returns>
    DWORD WINAPI Nui_ProcessThread();

	//--S [] 2013/05/16: Sang-Wook Lee
	std::string generateTimestamp() const;
	//--E [] 2013/05/16: Sang-Wook Lee

private:
    bool m_fUpdatingUi;
    TCHAR m_szAppTitle[256];  // Application title

	//--S [] 2012/06/09: Sang-Wook Lee
    bool m_bSaveAVI;
    bool m_bCaptureColorImage, m_bCaptureDepthImage;
	const double FPS_;
	const cv::Size COLOR_FRAME_SIZE_;  // for RGBA or IR image
	const cv::Size DEPTH_FRAME_SIZE_;
	TCHAR saveFilePath_[MAX_PATH];
	boost::scoped_ptr<cv::VideoWriter> rgbaVideoWriter_;  // for RGBA or IR image
	boost::scoped_ptr<cv::VideoWriter> depthVideoWriter_;
	std::ofstream depthBinStream_;
	std::ofstream skelBinStream_;
	cv::Mat colorFrame_;  // for RGBA or IR image
	cv::Mat depthFrame_;
	std::string saved_base_path_name_, saved_timestamp_;
	const std::string saved_base_file_name_;
	std::size_t saved_depth_frame_id_;
	//--E [] 2012/06/09

	//--S [] 2012/07/26: Sang-Wook Lee
	enum {
		RECORD_COLOR_AVI = 0x01, RECORD_DEPTH_AVI = 0x02, RECORD_DEPTH_ASCII = 0x04, RECORD_DEPTH_BINARY = 0x08, RECORD_SKELETON_BINARY = 0x10
	};
	const int recordType_;

	bool setUpRecording(const std::string &filename);
	void cleanUpRecording();
	//--E [] 2012/07/26

	// Current kinect
    INuiSensor *m_pNuiSensor;
    BSTR m_instanceId;

    // Skeletal drawing
    ID2D1HwndRenderTarget *m_pRenderTarget;
    ID2D1SolidColorBrush *m_pBrushJointTracked;
    ID2D1SolidColorBrush *m_pBrushJointInferred;
    ID2D1SolidColorBrush *m_pBrushBoneTracked;
    ID2D1SolidColorBrush *m_pBrushBoneInferred;
    D2D1_POINT_2F m_Points[NUI_SKELETON_POSITION_COUNT];

    // Draw devices
    DrawDevice *m_pDrawDepth;
    DrawDevice *m_pDrawRGBA;
    ID2D1Factory *m_pD2DFactory;

    // thread handling
    HANDLE m_hThNuiProcess;
    HANDLE m_hEvNuiProcessStop;
    
    HANDLE m_hNextDepthFrameEvent;
    HANDLE m_hNextRGBAFrameEvent;
    HANDLE m_hNextSkeletonEvent;

	HANDLE m_pDepthStreamHandle;
    HANDLE m_pRGBAStreamHandle;

	//--S [] 2013/05/16: Sang-Wook Lee
    DrawDevice *m_pDrawIR;
    HANDLE m_hNextIRFrameEvent;
    HANDLE m_pIRStreamHandle;

	const bool m_bUseIRImageInsteadOfRGBAImage;
	const bool m_bTurnInfraredEmitterOff;
    RGBQUAD *m_pTempIRBuffer;
	//--E [] 2013/05/16: Sang-Wook Lee

    HFONT m_hFontFPS;
    BYTE m_depthRGBX[640*480*4];
    DWORD m_LastSkeletonFoundTime;
    bool m_bScreenBlanked;
    int m_DepthFramesTotal;
    DWORD m_LastDepthFPStime;
    int m_LastDepthFramesTotal;
    int m_TrackedSkeletons;
    DWORD m_SkeletonTrackingFlags;
    DWORD m_DepthStreamFlags;

    DWORD m_StickySkeletonIds[NUI_SKELETON_MAX_TRACKED_COUNT];
};
