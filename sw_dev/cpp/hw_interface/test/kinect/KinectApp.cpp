#include "stdafx.h"
#include "KinectApp.h"
#include "resource.h"
#include <CommDlg.h>
#include <mmsystem.h>
#include <strsafe.h>
#include <cassert>


//#define __USE_DEPTH_IMAGE_320x240 1
#define __USE_DEPTH_IMAGE_640x480 1

//lookups for color tinting based on player index
static const int g_IntensityShiftByPlayerR[] = { 1, 2, 0, 2, 0, 0, 2, 0 };
static const int g_IntensityShiftByPlayerG[] = { 1, 2, 2, 0, 2, 0, 0, 1 };
static const int g_IntensityShiftByPlayerB[] = { 1, 0, 2, 2, 0, 2, 0, 2 };

static const float g_JointThickness = 3.0f;
static const float g_TrackedBoneThickness = 6.0f;
static const float g_InferredBoneThickness = 1.0f;

const int g_BytesPerPixel = 4;

const int g_ScreenWidth = 320;
const int g_ScreenHeight = 240;

enum _SV_TRACKED_SKELETONS
{
    SV_TRACKED_SKELETONS_DEFAULT = 0,
    SV_TRACKED_SKELETONS_NEAREST1,
    SV_TRACKED_SKELETONS_NEAREST2,
    SV_TRACKED_SKELETONS_STICKY1,
    SV_TRACKED_SKELETONS_STICKY2
} SV_TRACKED_SKELETONS;

enum _SV_TRACKING_MODE
{
    SV_TRACKING_MODE_DEFAULT = 0,
    SV_TRACKING_MODE_SEATED
} SV_TRACKING_MODE;

enum _SV_RANGE
{
    SV_RANGE_DEFAULT = 0,
    SV_RANGE_NEAR,
} SV_RANGE;

//-------------------------------------------------------------------
// Constructor
//-------------------------------------------------------------------
CKinectApp::CKinectApp()
: m_hInstance(NULL),
  m_bSaveFrames(true), FPS_(30), FRAME_SIZE_(640, 480), frame_(FRAME_SIZE_, CV_8UC3, cv::Scalar::all(0)),  //-- [] 2012/06/09: Sang-Wook Lee
  recordType_(RECORD_COLOR_IMAGE)  //-- [] 2012/07/26: Sang-Wook Lee
{
    ZeroMemory(m_szAppTitle, sizeof(m_szAppTitle));
    LoadString(m_hInstance, IDS_APP_TITLE, m_szAppTitle, _countof(m_szAppTitle));

    m_fUpdatingUi = false;
    Nui_Zero();

    // Init Direct2D
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &m_pD2DFactory);

	//--S [] 2012/07/26: Sang-Wook Lee
	if (m_bSaveFrames)
	{
		TCHAR path[MAX_PATH] = { _T('\0'), };
		GetCurrentDirectory(MAX_PATH, path);

		char filepath[MAX_PATH];
#if defined(UNICODE) || defined(_UNICODE)
		WideCharToMultiByte(CP_ACP, 0, path, MAX_PATH, filepath, MAX_PATH, NULL, NULL);
#else
		filepath = path;
#endif

		const std::string recordFilePath(std::string(filepath) + "\\kinect_record.avi");
		setUpRecording(recordFilePath);
	}
	//--E [] 2012/07/26
}

//-------------------------------------------------------------------
// Destructor
//-------------------------------------------------------------------
CKinectApp::~CKinectApp()
{
	//--S [] 2012/07/26: Sang-Wook Lee
	cleanUpRecording();
	//--E [] 2012/07/26

	// Clean up Direct2D
    SafeRelease(m_pD2DFactory);

    Nui_Zero();
    SysFreeString(m_instanceId);
}

void CKinectApp::ClearKinectComboBox()
{
    for (long i = 0; i < SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_GETCOUNT, 0, 0); ++i)
    {
        SysFreeString(reinterpret_cast<BSTR>(SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_GETITEMDATA, i, 0)));
    }
    SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_RESETCONTENT, 0, 0);
}

void CKinectApp::UpdateKinectComboBox()
{
    m_fUpdatingUi = true;
    ClearKinectComboBox();

    int numDevices = 0;
    HRESULT hr = NuiGetSensorCount(&numDevices);

    if (FAILED(hr))
    {
        return;
    }

    long selectedIndex = 0;
    for (int i = 0; i < numDevices; ++i)
    {
        INuiSensor *pNui = NULL;
        HRESULT hr = NuiCreateSensorByIndex(i, &pNui);
        if (SUCCEEDED(hr))
        {
            HRESULT status = pNui ? pNui->NuiStatus() : E_NUI_NOTCONNECTED;
            if (status == E_NUI_NOTCONNECTED)
            {
                pNui->Release();
                continue;
            }
            
            WCHAR kinectName[MAX_PATH];
            StringCchPrintfW(kinectName, _countof(kinectName), L"Kinect %d", i);
            long index = static_cast<long>(SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(kinectName)));
            SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_SETITEMDATA, index, reinterpret_cast<LPARAM>(pNui->NuiUniqueId()));
            if (m_pNuiSensor && pNui == m_pNuiSensor)
            {
                selectedIndex = index;
            }
            pNui->Release();
        }
    }

    SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_SETCURSEL, selectedIndex, 0);
    m_fUpdatingUi = false;
}

LRESULT CALLBACK CKinectApp::MessageRouter(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    CKinectApp *pThis = NULL;
    
    if (WM_INITDIALOG == uMsg)
    {
        pThis = reinterpret_cast<CKinectApp *>(lParam);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        NuiSetDeviceStatusCallback(&CKinectApp::Nui_StatusProcThunk, pThis);
    }
    else
    {
        pThis = reinterpret_cast<CKinectApp *>(::GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }

    if (NULL != pThis)
    {
        return pThis->WndProc(hwnd, uMsg, wParam, lParam);
    }

    return 0;
}

//-------------------------------------------------------------------
// WndProc
//
// Handle windows messages
//-------------------------------------------------------------------
LRESULT CALLBACK CKinectApp::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
        case WM_INITDIALOG:
        {
            // Clean state the class
            Nui_Zero();

            // Bind application window handle
            m_hWnd = hWnd;

            // Set the font for Frames Per Second display
            LOGFONT lf;
            GetObject((HFONT)GetStockObject(DEFAULT_GUI_FONT), sizeof(lf), &lf);
            lf.lfHeight *= 4;
            m_hFontFPS = CreateFontIndirect(&lf);
            SendDlgItemMessageW(hWnd, IDC_FPS, WM_SETFONT, (WPARAM)m_hFontFPS, 0);

            UpdateKinectComboBox();
            SendDlgItemMessageW(m_hWnd, IDC_CAMERAS, CB_SETCURSEL, 0, 0);

            TCHAR szComboText[512] = { 0 };

            // Fill combo box options for tracked skeletons

            LoadStringW(m_hInstance, IDS_TRACKEDSKELETONS_DEFAULT, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_TRACKEDSKELETONS_NEAREST1, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_TRACKEDSKELETONS_NEAREST2, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_TRACKEDSKELETONS_STICKY1, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_TRACKEDSKELETONS_STICKY2, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_SETCURSEL, 0, 0);
            // Fill combo box options for tracking mode

            LoadStringW(m_hInstance, IDS_TRACKINGMODE_DEFAULT, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKINGMODE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_TRACKINGMODE_SEATED, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKINGMODE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_TRACKINGMODE, CB_SETCURSEL, 0, 0);

            // Fill combo box options for range

            LoadStringW(m_hInstance, IDS_RANGE_DEFAULT, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_RANGE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            LoadStringW(m_hInstance, IDS_RANGE_NEAR, szComboText, _countof(szComboText));
            SendDlgItemMessageW(m_hWnd, IDC_RANGE, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(szComboText));

            SendDlgItemMessageW(m_hWnd, IDC_RANGE, CB_SETCURSEL, 0, 0);
        }
        break;

        case WM_SHOWWINDOW:
        {
            // Initialize and start NUI processing
            Nui_Init();
        }
        break;

        case WM_USER_UPDATE_FPS:
        {
            ::SetDlgItemInt(m_hWnd, static_cast<int>(wParam), static_cast<int>(lParam), FALSE);
        }
        break;

        case WM_USER_UPDATE_COMBO:
        {
            UpdateKinectComboBox();
        }
        break;

        case WM_COMMAND:
        {
            if (HIWORD(wParam) == CBN_SELCHANGE)
            {
                switch (LOWORD(wParam))
                {
                    case IDC_CAMERAS:
                    {
                        LRESULT index = ::SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_GETCURSEL, 0, 0);

                        // Don't reconnect as a result of updating the combo box
                        if (!m_fUpdatingUi)
                        {
                            Nui_UnInit();
                            Nui_Zero();
                            Nui_Init(reinterpret_cast<BSTR>(::SendDlgItemMessage(m_hWnd, IDC_CAMERAS, CB_GETITEMDATA, index, 0)));
                        }
                    }
                    break;

                    case IDC_TRACKEDSKELETONS:
                    {
                        LRESULT index = ::SendDlgItemMessageW(m_hWnd, IDC_TRACKEDSKELETONS, CB_GETCURSEL, 0, 0);
                        UpdateTrackedSkeletonSelection(static_cast<int>(index));
                    }
                    break;

                    case IDC_TRACKINGMODE:
                    {
                        LRESULT index = ::SendDlgItemMessageW(m_hWnd, IDC_TRACKINGMODE, CB_GETCURSEL, 0, 0);
                        UpdateTrackingMode(static_cast<int>(index));
                    }
                    break;

                    case IDC_RANGE:
                    {
                        LRESULT index = ::SendDlgItemMessageW(m_hWnd, IDC_RANGE, CB_GETCURSEL, 0, 0);
                        UpdateRange(static_cast<int>(index));
                    }
                    break;
                }
            }
            else if (HIWORD(wParam) == BN_CLICKED)
            {
                switch (LOWORD(wParam))
                {
					//--S [] 2012/06/09: Sang-Wook Lee
                    case IDC_SAVE_FRAMES:
                    {
						const bool checked = IsDlgButtonChecked(m_hWnd, IDC_SAVE_FRAMES) == BST_CHECKED;
                        if (checked)
                        {
							saveFilePath_[0] = _T('\0');
                            if (OnFileSave(m_hWnd, ID_FILE_SAVE, 0, 0) == TRUE)
							{
								cleanUpRecording();

								char filepath[MAX_PATH];
#if defined(UNICODE) || defined(_UNICODE)
								WideCharToMultiByte(CP_ACP, 0, saveFilePath_, MAX_PATH, filepath, MAX_PATH, NULL, NULL);
#else
								filepath = saveFilePath_;
#endif
								if (setUpRecording(filepath))
									m_bSaveFrames = true;
								else
								{
									MessageBox(NULL, TEXT("file I/O failed to open"), TEXT("Creation Error"), MB_ICONSTOP);
									m_bSaveFrames = false;
									cleanUpRecording();
								}
							}
							else
							{
								m_bSaveFrames = false;
								cleanUpRecording();
							}

							if (checked != m_bSaveFrames)
								CheckDlgButton(m_hWnd, IDC_SAVE_FRAMES, m_bSaveFrames ? BST_CHECKED : BST_UNCHECKED);
                        }
						else
						{
							m_bSaveFrames = false;
							cleanUpRecording();
						}
                    }
                    break;
					//--E [] 2012/06/09
                }
            }
        }
        break;

        // If the titlebar X is clicked destroy app
        case WM_CLOSE:
            DestroyWindow(hWnd);
            break;

        case WM_DESTROY:
            // Uninitialize NUI
            Nui_UnInit();

            // Other cleanup
            ClearKinectComboBox();
            DeleteObject(m_hFontFPS);

            // Quit the main message pump
            PostQuitMessage(0);
            break;
    }

    return FALSE;
}

bool CKinectApp::setUpRecording(const std::string &filepath)
{
	const std::string video_filepath(filepath);
	const std::string::size_type extPos = video_filepath.find_last_of('.');
	const std::string depth_filepath(video_filepath.substr(0, extPos + 1) + std::string("depth"));;
	const std::string skel_filepath(video_filepath.substr(0, extPos + 1) + std::string("skel"));;

	const bool isColor = true;
	videoWriter_.reset(new cv::VideoWriter(video_filepath, CV_FOURCC('D', 'I', 'V', 'X'), FPS_, FRAME_SIZE_, isColor));
	depthstream_.open(depth_filepath.c_str(), std::ios::trunc | std::ios::out | std::ios::binary);
	skelstream_.open(skel_filepath.c_str(), std::ios::trunc | std::ios::out | std::ios::binary);
	// TODO [add] >>

	return videoWriter_->isOpened() && depthstream_ && skelstream_;
}

void CKinectApp::cleanUpRecording()
{
	videoWriter_.reset();
	depthstream_.close();
	skelstream_.close();
}

//-------------------------------------------------------------------
// MessageBoxResource
//
// Display a MessageBox with a string table table loaded string
//-------------------------------------------------------------------
int CKinectApp::MessageBoxResource(UINT nID, UINT nType)
{
    static TCHAR szRes[512];

    LoadString(m_hInstance, nID, szRes, _countof(szRes));
    return MessageBox(m_hWnd, szRes, m_szAppTitle, nType);
}

//-------------------------------------------------------------------
// Nui_Zero
//
// Zero out member variables
//-------------------------------------------------------------------
void CKinectApp::Nui_Zero()
{
    SafeRelease(m_pNuiSensor);

    m_pRenderTarget = NULL;
    m_pBrushJointTracked = NULL;
    m_pBrushJointInferred = NULL;
    m_pBrushBoneTracked = NULL;
    m_pBrushBoneInferred = NULL;
    ZeroMemory(m_Points, sizeof(m_Points));

    m_hNextDepthFrameEvent = NULL;
    m_hNextColorFrameEvent = NULL;
    m_hNextSkeletonEvent = NULL;
    m_pDepthStreamHandle = NULL;
    m_pVideoStreamHandle = NULL;
    m_hThNuiProcess = NULL;
    m_hEvNuiProcessStop = NULL;
    m_LastSkeletonFoundTime = 0;
    m_bScreenBlanked = false;
    m_DepthFramesTotal = 0;
    m_LastDepthFPStime = 0;
    m_LastDepthFramesTotal = 0;
    m_pDrawDepth = NULL;
    m_pDrawColor = NULL;
    m_TrackedSkeletons = 0;
    m_SkeletonTrackingFlags = NUI_SKELETON_TRACKING_FLAG_ENABLE_IN_NEAR_RANGE;
    m_DepthStreamFlags = 0;
    ZeroMemory(m_StickySkeletonIds, sizeof(m_StickySkeletonIds));
}

void CALLBACK CKinectApp::Nui_StatusProcThunk(HRESULT hrStatus, const OLECHAR *instanceName, const OLECHAR *uniqueDeviceName, void *pUserData)
{
    reinterpret_cast<CKinectApp *>(pUserData)->Nui_StatusProc(hrStatus, instanceName, uniqueDeviceName);
}

//-------------------------------------------------------------------
// Nui_StatusProc
//
// Callback to handle Kinect status changes
//-------------------------------------------------------------------
void CALLBACK CKinectApp::Nui_StatusProc(HRESULT hrStatus, const OLECHAR *instanceName, const OLECHAR *uniqueDeviceName)
{
    // Update UI
    PostMessageW(m_hWnd, WM_USER_UPDATE_COMBO, 0, 0);

    if (SUCCEEDED(hrStatus))
    {
        if (S_OK == hrStatus)
        {
            if (m_instanceId && 0 == wcscmp(instanceName, m_instanceId))
            {
                Nui_Init(m_instanceId);
            }
            else if (!m_pNuiSensor)
            {
                Nui_Init();
            }
        }
    }
    else
    {
        if (m_instanceId && 0 == wcscmp(instanceName, m_instanceId))
        {
            Nui_UnInit();
            Nui_Zero();
        }
    }
}

//-------------------------------------------------------------------
// Nui_Init
//
// Initialize Kinect by instance name
//-------------------------------------------------------------------
HRESULT CKinectApp::Nui_Init(OLECHAR *instanceName)
{
    // Generic creation failure
    if (NULL == instanceName)
    {
        MessageBoxResource(IDS_ERROR_NUICREATE, MB_OK | MB_ICONHAND);
        return E_FAIL;
    }

    HRESULT hr = NuiCreateSensorById(instanceName, &m_pNuiSensor);
    
    // Generic creation failure
    if (FAILED(hr))
    {
        MessageBoxResource(IDS_ERROR_NUICREATE, MB_OK | MB_ICONHAND);
        return hr;
    }

    SysFreeString(m_instanceId);

    m_instanceId = m_pNuiSensor->NuiDeviceConnectionId();

    return Nui_Init();
}

//-------------------------------------------------------------------
// Nui_Init
//
// Initialize Kinect
//-------------------------------------------------------------------
HRESULT CKinectApp::Nui_Init()
{
    HRESULT  hr;
    bool     result;

    if (!m_pNuiSensor)
    {
        HRESULT hr = NuiCreateSensorByIndex(0, &m_pNuiSensor);

        if (FAILED(hr))
        {
            return hr;
        }

        SysFreeString(m_instanceId);

        m_instanceId = m_pNuiSensor->NuiDeviceConnectionId();
    }

    m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
    m_hNextSkeletonEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

    // reset the tracked skeletons, range, and tracking mode
    SendDlgItemMessage(m_hWnd, IDC_TRACKEDSKELETONS, CB_SETCURSEL, 0, 0);
    SendDlgItemMessage(m_hWnd, IDC_TRACKINGMODE, CB_SETCURSEL, 0, 0);
    SendDlgItemMessage(m_hWnd, IDC_RANGE, CB_SETCURSEL, 0, 0);

    EnsureDirect2DResources();

    m_pDrawDepth = new DrawDevice();
#if __USE_DEPTH_IMAGE_320x240
	result = m_pDrawDepth->Initialize(GetDlgItem(m_hWnd, IDC_DEPTHVIEWER), m_pD2DFactory, 320, 240, 320 * 4);
#elif __USE_DEPTH_IMAGE_640x480
    result = m_pDrawDepth->Initialize(GetDlgItem(m_hWnd, IDC_DEPTHVIEWER), m_pD2DFactory, 640, 480, 640 * 4);
#else
	result = false;
#endif
	if (!result)
    {
        MessageBoxResource(IDS_ERROR_DRAWDEVICE, MB_OK | MB_ICONHAND);
        return E_FAIL;
    }

    m_pDrawColor = new DrawDevice();
    result = m_pDrawColor->Initialize(GetDlgItem(m_hWnd, IDC_VIDEOVIEW), m_pD2DFactory, 640, 480, 640 * 4);
    if (!result)
    {
        MessageBoxResource(IDS_ERROR_DRAWDEVICE, MB_OK | MB_ICONHAND);
        return E_FAIL;
    }
    
    DWORD nuiFlags = NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX | NUI_INITIALIZE_FLAG_USES_SKELETON | NUI_INITIALIZE_FLAG_USES_COLOR;
    hr = m_pNuiSensor->NuiInitialize(nuiFlags);
    if (E_NUI_SKELETAL_ENGINE_BUSY == hr)
    {
        nuiFlags = NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR;
        hr = m_pNuiSensor->NuiInitialize(nuiFlags);
    }
    if (FAILED(hr))
    {
        if (E_NUI_DEVICE_IN_USE == hr)
        {
            MessageBoxResource(IDS_ERROR_IN_USE, MB_OK | MB_ICONHAND);
        }
        else
        {
            MessageBoxResource(IDS_ERROR_NUIINIT, MB_OK | MB_ICONHAND);
        }
        return hr;
    }

    if (HasSkeletalEngine(m_pNuiSensor))
    {
        hr = m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonEvent, 0);
        if (FAILED(hr))
        {
            MessageBoxResource(IDS_ERROR_SKELETONTRACKING, MB_OK | MB_ICONHAND);
            return hr;
        }
    }

    hr = m_pNuiSensor->NuiImageStreamOpen(
        NUI_IMAGE_TYPE_COLOR,
        NUI_IMAGE_RESOLUTION_640x480,
        0,
        2,
        m_hNextColorFrameEvent,
        &m_pVideoStreamHandle
	);
    if (FAILED(hr))
    {
        MessageBoxResource(IDS_ERROR_VIDEOSTREAM, MB_OK | MB_ICONHAND);
        return hr;
    }

    hr = m_pNuiSensor->NuiImageStreamOpen(
        HasSkeletalEngine(m_pNuiSensor) ? NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX : NUI_IMAGE_TYPE_DEPTH,
#if __USE_DEPTH_IMAGE_320x240
		NUI_IMAGE_RESOLUTION_320x240,
#elif __USE_DEPTH_IMAGE_640x480
        NUI_IMAGE_RESOLUTION_640x480,
#endif
		0,
        2,
        m_hNextDepthFrameEvent,
        &m_pDepthStreamHandle
	);
    if (FAILED(hr))
    {
        MessageBoxResource(IDS_ERROR_DEPTHSTREAM, MB_OK | MB_ICONHAND);
        return hr;
    }

    // Start the Nui processing thread
    m_hEvNuiProcessStop = CreateEvent(NULL, FALSE, FALSE, NULL);
    m_hThNuiProcess = CreateThread(NULL, 0, Nui_ProcessThread, this, 0, NULL);

    return hr;
}

//-------------------------------------------------------------------
// Nui_UnInit
//
// Uninitialize Kinect
//-------------------------------------------------------------------
void CKinectApp::Nui_UnInit()
{
    // Stop the Nui processing thread
    if (NULL != m_hEvNuiProcessStop)
    {
        // Signal the thread
        SetEvent(m_hEvNuiProcessStop);

        // Wait for thread to stop
        if (NULL != m_hThNuiProcess)
        {
            WaitForSingleObject(m_hThNuiProcess, INFINITE);
            CloseHandle(m_hThNuiProcess);
        }
        CloseHandle(m_hEvNuiProcessStop);
    }

    if (m_pNuiSensor)
    {
        m_pNuiSensor->NuiShutdown();
    }
    if (m_hNextSkeletonEvent && (m_hNextSkeletonEvent != INVALID_HANDLE_VALUE))
    {
        CloseHandle(m_hNextSkeletonEvent);
        m_hNextSkeletonEvent = NULL;
    }
    if (m_hNextDepthFrameEvent && (m_hNextDepthFrameEvent != INVALID_HANDLE_VALUE))
    {
        CloseHandle(m_hNextDepthFrameEvent);
        m_hNextDepthFrameEvent = NULL;
    }
    if (m_hNextColorFrameEvent && (m_hNextColorFrameEvent != INVALID_HANDLE_VALUE))
    {
        CloseHandle(m_hNextColorFrameEvent);
        m_hNextColorFrameEvent = NULL;
    }

    SafeRelease(m_pNuiSensor);

    // clean up graphics
    delete m_pDrawDepth;
    m_pDrawDepth = NULL;

    delete m_pDrawColor;
    m_pDrawColor = NULL;    

    DiscardDirect2DResources();
}

DWORD WINAPI CKinectApp::Nui_ProcessThread(LPVOID pParam)
{
    CKinectApp *pthis = (CKinectApp *)pParam;
    return pthis->Nui_ProcessThread();
}

//-------------------------------------------------------------------
// Nui_ProcessThread
//
// Thread to handle Kinect processing
//-------------------------------------------------------------------
DWORD WINAPI CKinectApp::Nui_ProcessThread()
{
    const int numEvents = 4;
    HANDLE hEvents[numEvents] = { m_hEvNuiProcessStop, m_hNextDepthFrameEvent, m_hNextColorFrameEvent, m_hNextSkeletonEvent };
    int nEventIdx;
    DWORD t;

    m_LastDepthFPStime = timeGetTime();

    //blank the skeleton display on startup
    m_LastSkeletonFoundTime = 0;

    // Main thread loop
    bool continueProcessing = true;
    while (continueProcessing)
    {
        // Wait for any of the events to be signalled
        nEventIdx = WaitForMultipleObjects(numEvents, hEvents, FALSE, 100);

        // Timed out, continue
        if (WAIT_TIMEOUT == nEventIdx)
            continue;

        // stop event was signalled 
        if (WAIT_OBJECT_0 == nEventIdx)
        {
            continueProcessing = false;
            break;
        }

        // Wait for each object individually with a 0 timeout to make sure to process all signalled objects if multiple objects were signalled this loop iteration

        // In situations where perfect correspondance between color/depth/skeleton is essential, a priority queue should be used to service the item which has been updated the longest ago

        if (WAIT_OBJECT_0 == WaitForSingleObject(m_hNextDepthFrameEvent, 0))
        {
            //only increment frame count if a frame was successfully drawn
            if (Nui_GotDepthAlert())
                ++m_DepthFramesTotal;
        }

        if (WAIT_OBJECT_0 == WaitForSingleObject(m_hNextColorFrameEvent, 0))
            Nui_GotColorAlert();

        if ( WAIT_OBJECT_0 == WaitForSingleObject(m_hNextSkeletonEvent, 0))
            Nui_GotSkeletonAlert();

        // Once per second, display the depth FPS
        t = timeGetTime();
        if ((t - m_LastDepthFPStime) > 1000)
        {
            const int fps = ((m_DepthFramesTotal - m_LastDepthFramesTotal) * 1000 + 500) / (t - m_LastDepthFPStime);
            PostMessageW(m_hWnd, WM_USER_UPDATE_FPS, IDC_FPS, fps);
            m_LastDepthFramesTotal = m_DepthFramesTotal;
            m_LastDepthFPStime = t;
        }

        // Blank the skeleton panel if we haven't found a skeleton recently
        if ((t - m_LastSkeletonFoundTime) > 300)
        {
            if (!m_bScreenBlanked)
            {
                Nui_BlankSkeletonScreen();
                m_bScreenBlanked = true;
            }
        }
    }

    return 0;
}

//-------------------------------------------------------------------
// Nui_GotDepthAlert
//
// Handle new color data
//-------------------------------------------------------------------
bool CKinectApp::Nui_GotColorAlert()
{
    NUI_IMAGE_FRAME imageFrame;
    HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pVideoStreamHandle, 0, &imageFrame);
    if (FAILED(hr))
    {
        return false;
    }

    bool processedFrame = true;
    INuiFrameTexture *pTexture = imageFrame.pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    pTexture->LockRect(0, &LockedRect, NULL, 0);
    if (0 != LockedRect.Pitch)
    {
        m_pDrawColor->Draw(static_cast<BYTE *>(LockedRect.pBits), LockedRect.size);

		//--S [] 2012/06/09: Sang-Wook Lee
		if (m_bSaveFrames && videoWriter_ && RECORD_COLOR_IMAGE == (RECORD_COLOR_IMAGE & recordType_))
		{
#if 1
			const cv::Mat frameBGRA(FRAME_SIZE_, CV_8UC4, static_cast<void *>(LockedRect.pBits));
			cv::cvtColor(frameBGRA, frame_, CV_BGRA2BGR, 3);
			*videoWriter_ << frame_;

			OutputDebugString(_T("."));
#else
			IplImage *frameBGRA = cvCreateImage(cvSize(FRAME_SIZE_.width, FRAME_SIZE_.height), IPL_DEPTH_8U, 4);
			//strcpy(frameBGRA->colorModel, "BGRA");
			//strcpy(frameBGRA->channelSeq, "BGRA");
			frameBGRA->widthStep = frameBGRA->width * 4;
			frameBGRA->imageSize = frameBGRA->widthStep * frameBGRA->height;
			frameBGRA->imageData = (char *)(LockedRect.pBits);
			IplImage *frameBGR = cvCreateImage(cvSize(FRAME_SIZE_.width, FRAME_SIZE_.height), IPL_DEPTH_8U, 3);

			cvCvtColor(frameBGRA, frameBGR, CV_BGRA2BGR);
			*videoWriter_ << cv::Mat(frameBGR);
			
			cvReleaseImage(&frameBGR);
			cvReleaseImage(&frameBGRA);
#endif
		}
		//--E [] 2012/06/09
    }
    else
    {
        OutputDebugString(L"Buffer length of received texture is bogus\r\n");
        processedFrame = false;
    }

    pTexture->UnlockRect(0);

    m_pNuiSensor->NuiImageStreamReleaseFrame(m_pVideoStreamHandle, &imageFrame);

    return processedFrame;
}

//-------------------------------------------------------------------
// Nui_GotDepthAlert
//
// Handle new depth data
//-------------------------------------------------------------------
bool CKinectApp::Nui_GotDepthAlert()
{
    NUI_IMAGE_FRAME imageFrame;
    HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
    if (FAILED(hr))
    {
        return false;
    }

    bool processedFrame = true;
    INuiFrameTexture *pTexture = imageFrame.pFrameTexture;
    NUI_LOCKED_RECT LockedRect;
    pTexture->LockRect(0, &LockedRect, NULL, 0);
    if (0 != LockedRect.Pitch)
    {
        DWORD frameWidth, frameHeight;
        NuiImageResolutionToSize(imageFrame.eResolution, frameWidth, frameHeight);
        
        // draw the bits to the bitmap
        BYTE *rgbrun = m_depthRGBX;
        const USHORT *pBufferRun = (const USHORT *)LockedRect.pBits;
        // end pixel is start + width * height - 1
        const USHORT *pBufferEnd = pBufferRun + (frameWidth * frameHeight);

        assert(frameWidth * frameHeight * g_BytesPerPixel <= ARRAYSIZE(m_depthRGBX));

		// for display
        while (pBufferRun < pBufferEnd)
        {
            USHORT depth = *pBufferRun;
            USHORT realDepth = NuiDepthPixelToDepth(depth);
            USHORT player = NuiDepthPixelToPlayerIndex(depth);

            // transform 13-bit depth information into an 8-bit intensity appropriate
            // for display (we disregard information in most significant bit)
            BYTE intensity = static_cast<BYTE>(~(realDepth >> 4));

            // tint the intensity by dividing by per-player values
            *(rgbrun++) = intensity >> g_IntensityShiftByPlayerB[player];
            *(rgbrun++) = intensity >> g_IntensityShiftByPlayerG[player];
            *(rgbrun++) = intensity >> g_IntensityShiftByPlayerR[player];

            // no alpha information, skip the last byte
            ++rgbrun;

            ++pBufferRun;
        }

        m_pDrawDepth->Draw(m_depthRGBX, frameWidth * frameHeight * g_BytesPerPixel);

		//--S [] 2012/06/09: Sang-Wook Lee
		if (m_bSaveFrames && depthstream_ && RECORD_DEPTH_IMAGE == (RECORD_DEPTH_IMAGE & recordType_))
		{
			depthstream_.write(reinterpret_cast<char *>(LockedRect.pBits), sizeof(USHORT) * frameWidth * frameHeight);
		}
		//--E [] 2012/06/09
    }
    else
    {
        OutputDebugString(L"Buffer length of received texture is bogus\r\n");
        processedFrame = false;
    }

    pTexture->UnlockRect(0);

    m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);

    return processedFrame;
}

void CKinectApp::Nui_BlankSkeletonScreen()
{
    m_pRenderTarget->BeginDraw();
    m_pRenderTarget->Clear();
    m_pRenderTarget->EndDraw();
}

void CKinectApp::Nui_DrawBone(const NUI_SKELETON_DATA &skel, NUI_SKELETON_POSITION_INDEX bone0, NUI_SKELETON_POSITION_INDEX bone1)
{
    NUI_SKELETON_POSITION_TRACKING_STATE bone0State = skel.eSkeletonPositionTrackingState[bone0];
    NUI_SKELETON_POSITION_TRACKING_STATE bone1State = skel.eSkeletonPositionTrackingState[bone1];

    // If we can't find either of these joints, exit
    if (NUI_SKELETON_POSITION_NOT_TRACKED == bone0State || NUI_SKELETON_POSITION_NOT_TRACKED == bone1State)
        return;
    
    // Don't draw if both points are inferred
    if (NUI_SKELETON_POSITION_INFERRED == bone0State && NUI_SKELETON_POSITION_INFERRED == bone1State)
        return;

    // We assume all drawn bones are inferred unless BOTH joints are tracked
    if (NUI_SKELETON_POSITION_TRACKED == bone0State && NUI_SKELETON_POSITION_TRACKED == bone1State)
        m_pRenderTarget->DrawLine(m_Points[bone0], m_Points[bone1], m_pBrushBoneTracked, g_TrackedBoneThickness);
    else
        m_pRenderTarget->DrawLine(m_Points[bone0], m_Points[bone1], m_pBrushBoneInferred, g_InferredBoneThickness);
}

void CKinectApp::Nui_DrawSkeleton(const NUI_SKELETON_DATA &skel, int windowWidth, int windowHeight)
{
    for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
        m_Points[i] = SkeletonToScreen(skel.SkeletonPositions[i], windowWidth, windowHeight);

    // Render Torso
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_HEAD, NUI_SKELETON_POSITION_SHOULDER_CENTER);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SHOULDER_RIGHT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_CENTER, NUI_SKELETON_POSITION_SPINE);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SPINE, NUI_SKELETON_POSITION_HIP_CENTER);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_HIP_CENTER, NUI_SKELETON_POSITION_HIP_RIGHT);

    // Left Arm
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_LEFT, NUI_SKELETON_POSITION_ELBOW_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_ELBOW_LEFT, NUI_SKELETON_POSITION_WRIST_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_WRIST_LEFT, NUI_SKELETON_POSITION_HAND_LEFT);

    // Right Arm
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_SHOULDER_RIGHT, NUI_SKELETON_POSITION_ELBOW_RIGHT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_ELBOW_RIGHT, NUI_SKELETON_POSITION_WRIST_RIGHT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_WRIST_RIGHT, NUI_SKELETON_POSITION_HAND_RIGHT);

    // Left Leg
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_HIP_LEFT, NUI_SKELETON_POSITION_KNEE_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_KNEE_LEFT, NUI_SKELETON_POSITION_ANKLE_LEFT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_ANKLE_LEFT, NUI_SKELETON_POSITION_FOOT_LEFT);

    // Right Leg
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_HIP_RIGHT, NUI_SKELETON_POSITION_KNEE_RIGHT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_KNEE_RIGHT, NUI_SKELETON_POSITION_ANKLE_RIGHT);
    Nui_DrawBone(skel, NUI_SKELETON_POSITION_ANKLE_RIGHT, NUI_SKELETON_POSITION_FOOT_RIGHT);
    
    // Draw the joints in a different color
    for (int i = 0; i < NUI_SKELETON_POSITION_COUNT; ++i)
    {
        const D2D1_ELLIPSE ellipse = D2D1::Ellipse(m_Points[i], g_JointThickness, g_JointThickness);

        if (NUI_SKELETON_POSITION_INFERRED == skel.eSkeletonPositionTrackingState[i])
            m_pRenderTarget->DrawEllipse(ellipse, m_pBrushJointInferred);
        else if (NUI_SKELETON_POSITION_TRACKED == skel.eSkeletonPositionTrackingState[i])
            m_pRenderTarget->DrawEllipse(ellipse, m_pBrushJointTracked);
    }
}

void CKinectApp::UpdateTrackedSkeletons(const NUI_SKELETON_FRAME &skel)
{
    DWORD nearestIDs[2] = { 0, 0 };
    USHORT nearestDepths[2] = { NUI_IMAGE_DEPTH_MAXIMUM, NUI_IMAGE_DEPTH_MAXIMUM };

    // Purge old sticky skeleton IDs, if the user has left the frame, etc
    bool stickyID0Found = false;
    bool stickyID1Found = false;
    for (int i = 0 ; i < NUI_SKELETON_COUNT; ++i)
    {
        NUI_SKELETON_TRACKING_STATE trackingState = skel.SkeletonData[i].eTrackingState;

        if (NUI_SKELETON_TRACKED == trackingState || NUI_SKELETON_POSITION_ONLY == trackingState)
        {
            if (skel.SkeletonData[i].dwTrackingID == m_StickySkeletonIds[0])
                stickyID0Found = true;
            else if (skel.SkeletonData[i].dwTrackingID == m_StickySkeletonIds[1])
                stickyID1Found = true;
        }
    }

    if (!stickyID0Found && stickyID1Found)
    {
        m_StickySkeletonIds[0] = m_StickySkeletonIds[1];
        m_StickySkeletonIds[1] = 0;
    }
    else if (!stickyID0Found)
    {
        m_StickySkeletonIds[0] = 0;
    }
    else if (!stickyID1Found)
    {
        m_StickySkeletonIds[1] = 0;
    }

    // Calculate nearest and sticky skeletons
    for (int i = 0 ; i < NUI_SKELETON_COUNT; ++i)
    {
        NUI_SKELETON_TRACKING_STATE trackingState = skel.SkeletonData[i].eTrackingState;

        if (NUI_SKELETON_TRACKED == trackingState || NUI_SKELETON_POSITION_ONLY == trackingState)
        {
            // Save SkeletonIds for sticky mode if there's none already saved
            if (0 == m_StickySkeletonIds[0] && m_StickySkeletonIds[1] != skel.SkeletonData[i].dwTrackingID)
                m_StickySkeletonIds[0] = skel.SkeletonData[i].dwTrackingID;
            else if (0 == m_StickySkeletonIds[1] && m_StickySkeletonIds[0] != skel.SkeletonData[i].dwTrackingID)
                m_StickySkeletonIds[1] = skel.SkeletonData[i].dwTrackingID;

            LONG x, y;
            USHORT depth;

            // calculate the skeleton's position on the screen
            NuiTransformSkeletonToDepthImage(skel.SkeletonData[i].Position, &x, &y, &depth);

            if (depth < nearestDepths[0])
            {
                nearestDepths[1] = nearestDepths[0];
                nearestIDs[1] = nearestIDs[0];

                nearestDepths[0] = depth;
                nearestIDs[0] = skel.SkeletonData[i].dwTrackingID;
            }
            else if (depth < nearestDepths[1])
            {
                nearestDepths[1] = depth;
                nearestIDs[1] = skel.SkeletonData[i].dwTrackingID;
            }
        }
    }

    if (SV_TRACKED_SKELETONS_NEAREST1 == m_TrackedSkeletons || SV_TRACKED_SKELETONS_NEAREST2 == m_TrackedSkeletons)
    {
        // Only track the closest single skeleton in nearest 1 mode
        if (SV_TRACKED_SKELETONS_NEAREST1 == m_TrackedSkeletons)
        {
            nearestIDs[1] = 0;
        }
        m_pNuiSensor->NuiSkeletonSetTrackedSkeletons(nearestIDs);
    }

    if (SV_TRACKED_SKELETONS_STICKY1 == m_TrackedSkeletons || SV_TRACKED_SKELETONS_STICKY2 == m_TrackedSkeletons)
    {
        DWORD stickyIDs[2] = { m_StickySkeletonIds[0], m_StickySkeletonIds[1] };

        // Only track a single skeleton in sticky 1 mode
        if (SV_TRACKED_SKELETONS_STICKY1 == m_TrackedSkeletons)
        {
            stickyIDs[1] = 0;
        }
        m_pNuiSensor->NuiSkeletonSetTrackedSkeletons(stickyIDs);
    }
}

D2D1_POINT_2F CKinectApp::SkeletonToScreen(Vector4 skeletonPoint, int width, int height)
{
    LONG x, y;
    USHORT depth;

    // calculate the skeleton's position on the screen
    // NuiTransformSkeletonToDepthImage returns coordinates in NUI_IMAGE_RESOLUTION_320x240 space
    NuiTransformSkeletonToDepthImage(skeletonPoint, &x, &y, &depth);

    float screenPointX = static_cast<float>(x * width) / g_ScreenWidth;
    float screenPointY = static_cast<float>(y * height) / g_ScreenHeight;

    return D2D1::Point2F(screenPointX, screenPointY);
}

//-------------------------------------------------------------------
// Nui_GotSkeletonAlert
//
// Handle new skeleton data
//-------------------------------------------------------------------
bool CKinectApp::Nui_GotSkeletonAlert()
{
    NUI_SKELETON_FRAME SkeletonFrame = { 0 };
	HRESULT hr = m_pNuiSensor->NuiSkeletonGetNextFrame(0, &SkeletonFrame);

    bool foundSkeleton = false;
    if (SUCCEEDED(hr))
    {
        for (int i = 0; i < NUI_SKELETON_COUNT; ++i)
        {
            NUI_SKELETON_TRACKING_STATE trackingState = SkeletonFrame.SkeletonData[i].eTrackingState;
            if (NUI_SKELETON_TRACKED == trackingState || NUI_SKELETON_POSITION_ONLY == trackingState)
                foundSkeleton = true;
        }
    }
    // no skeletons!
    if (!foundSkeleton)
        return true;

    // smooth out the skeleton data
    hr = m_pNuiSensor->NuiTransformSmooth(&SkeletonFrame, NULL);
    if (FAILED(hr))
        return false;

    // we found a skeleton, re-start the skeletal timer
    m_bScreenBlanked = false;
    m_LastSkeletonFoundTime = timeGetTime();

    // Endure Direct2D is ready to draw
    hr = EnsureDirect2DResources();
    if (FAILED(hr))
        return false;

    m_pRenderTarget->BeginDraw();
    m_pRenderTarget->Clear();
    
    RECT rct;
    GetClientRect(GetDlgItem(m_hWnd, IDC_SKELETALVIEW), &rct);
    int width = rct.right;
    int height = rct.bottom;

    for (int i = 0 ; i < NUI_SKELETON_COUNT; ++i)
    {
        NUI_SKELETON_TRACKING_STATE trackingState = SkeletonFrame.SkeletonData[i].eTrackingState;

        if (NUI_SKELETON_TRACKED == trackingState)
        {
            // We're tracking the skeleton, draw it
            Nui_DrawSkeleton( SkeletonFrame.SkeletonData[i], width, height );
        }
        else if (NUI_SKELETON_POSITION_ONLY == trackingState)
        {
            // we've only received the center point of the skeleton, draw that
            const D2D1_ELLIPSE ellipse = D2D1::Ellipse(
                SkeletonToScreen(SkeletonFrame.SkeletonData[i].Position, width, height),
                g_JointThickness,
                g_JointThickness
            );

            m_pRenderTarget->DrawEllipse(ellipse, m_pBrushJointTracked);
        }
    }

    hr = m_pRenderTarget->EndDraw();

    UpdateTrackedSkeletons(SkeletonFrame);

    // Device lost, need to recreate the render target
    // We'll dispose it now and retry drawing
    if (D2DERR_RECREATE_TARGET == hr)
    {
        hr = S_OK;
        DiscardDirect2DResources();
        return false;
    }

    return true;
}

void CKinectApp::UpdateTrackedSkeletonSelection(int mode)
{
    m_TrackedSkeletons = mode;

    UpdateSkeletonTrackingFlag(
        NUI_SKELETON_TRACKING_FLAG_TITLE_SETS_TRACKED_SKELETONS,
        (SV_TRACKED_SKELETONS_DEFAULT != mode)
	);
}

void CKinectApp::UpdateTrackingMode(int mode)
{
    UpdateSkeletonTrackingFlag(
        NUI_SKELETON_TRACKING_FLAG_ENABLE_SEATED_SUPPORT,
        (SV_TRACKING_MODE_SEATED == mode)
	);
}

void CKinectApp::UpdateRange(int mode)
{
    UpdateDepthStreamFlag(
        NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE,
        (SV_RANGE_DEFAULT != mode)
	);
}

void CKinectApp::UpdateSkeletonTrackingFlag(DWORD flag, bool value)
{
    DWORD newFlags = m_SkeletonTrackingFlags;

    if (value)
        newFlags |= flag;
    else
        newFlags &= ~flag;

    if (NULL != m_pNuiSensor && newFlags != m_SkeletonTrackingFlags)
    {
        if (!HasSkeletalEngine(m_pNuiSensor))
            MessageBoxResource(IDS_ERROR_SKELETONTRACKING, MB_OK | MB_ICONHAND);

        m_SkeletonTrackingFlags = newFlags;

        HRESULT hr = m_pNuiSensor->NuiSkeletonTrackingEnable(m_hNextSkeletonEvent, m_SkeletonTrackingFlags);

        if (FAILED(hr))
            MessageBoxResource(IDS_ERROR_SKELETONTRACKING, MB_OK | MB_ICONHAND);
    }
}

void CKinectApp::UpdateDepthStreamFlag(DWORD flag, bool value)
{
    DWORD newFlags = m_DepthStreamFlags;

    if (value)
        newFlags |= flag;
    else
        newFlags &= ~flag;

    if (NULL != m_pNuiSensor && newFlags != m_DepthStreamFlags)
    {
        m_DepthStreamFlags = newFlags;
        m_pNuiSensor->NuiImageStreamSetImageFrameFlags(m_pDepthStreamHandle, m_DepthStreamFlags);
    }
}

HRESULT CKinectApp::EnsureDirect2DResources()
{
    HRESULT hr = S_OK;

    if (!m_pRenderTarget)
    {
        RECT rc;
        GetWindowRect(GetDlgItem(m_hWnd, IDC_SKELETALVIEW), &rc);  
    
        int width = rc.right - rc.left;
        int height = rc.bottom - rc.top;
        D2D1_SIZE_U size = D2D1::SizeU(width, height);
        D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties();
        rtProps.pixelFormat = D2D1::PixelFormat( DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE);
        rtProps.usage = D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE;

        // Create a Hwnd render target, in order to render to the window set in initialize
        hr = m_pD2DFactory->CreateHwndRenderTarget(
            rtProps,
            D2D1::HwndRenderTargetProperties(GetDlgItem( m_hWnd, IDC_SKELETALVIEW), size),
            &m_pRenderTarget
		);
        if (FAILED(hr))
        {
            MessageBoxResource(IDS_ERROR_DRAWDEVICE, MB_OK | MB_ICONHAND);
            return E_FAIL;
        }

        //light green
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(68, 192, 68), &m_pBrushJointTracked);

        //yellow
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(255, 255, 0), &m_pBrushJointInferred);

        //green
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(0, 128, 0), &m_pBrushBoneTracked);

        //gray
        m_pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(128, 128, 128), &m_pBrushBoneInferred);
    }

    return hr;
}

void CKinectApp::DiscardDirect2DResources()
{
    SafeRelease(m_pRenderTarget);

    SafeRelease(m_pBrushJointTracked);
    SafeRelease(m_pBrushJointInferred);
    SafeRelease(m_pBrushBoneTracked);
    SafeRelease(m_pBrushBoneInferred);
}

//--S [] 2012/06/09: Sang-Wook Lee
LPOFNHOOKPROC HookProcCenterDialog(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{	
	switch (message)
	{
	case WM_INITDIALOG:
		{
			// center the dialog
			HWND hWnd = GetParent(hDlg);
			RECT r1, r2;
			GetClientRect(hWnd, &r1);
			GetWindowRect(hDlg, &r2);
			POINT pt;
			pt.x = (r1.right - r1.left)/2 - (r2.right - r2.left)/2;
			pt.y = (r1.bottom - r1.top)/2 - (r2.bottom - r2.top)/2;
			ClientToScreen(hWnd, &pt);
			SetWindowPos(hDlg, HWND_TOP, pt.x, pt.y, 0, 0, SWP_NOSIZE);

			return FALSE;
		}
	}

	return 0;
}

LRESULT CALLBACK CKinectApp::OnFileOpen(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// here we place the file path 
	TCHAR szFile[MAX_PATH] = { 0 };

	// file extensions filter
	TCHAR *szFilter = TEXT("AVI Files (*.avi)\0*.avi\0All Files\0*.*\0\0");
	//TCHAR *szFilter = TEXT("AVI Files (*.avi)\0*.avi\0MPEG Files (*.mpg)\0*.mpg\0All Files\0*.*\0\0");

	// query the current folder
	TCHAR szCurDir[MAX_PATH];
	::GetCurrentDirectory(MAX_PATH - 1, szCurDir);

	// structure used by the standard file dialog
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(OPENFILENAME));

	// dialog parameters
	ofn.lStructSize	= sizeof(OPENFILENAME);

	// window which owns the dialog
	ofn.hwndOwner = GetParent(hWnd);

	ofn.lpstrFilter	= szFilter;
	// the filters string index (begins with 1)
	ofn.nFilterIndex = 1;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrDefExt = _T("avi");
	// dialog caption
	ofn.lpstrTitle	= _T("Open");
	ofn.nMaxFileTitle = sizeof(ofn.lpstrTitle);

	ofn.lpfnHook = HookProcCenterDialog((HWND)ofn.hwndOwner, ID_FILE_OPEN, 0, 0);

	// dialog style 
	ofn.Flags = OFN_ENABLEHOOK | OFN_EXPLORER;

	// create and open the dialog (retuns 0 on failure)
	if (GetOpenFileName(&ofn))
	{
		// try to open the file (which must exist)
		HANDLE hFile = CreateFile(
			ofn.lpstrFile, GENERIC_READ,
			FILE_SHARE_READ, 0, OPEN_EXISTING,
			FILE_ATTRIBUTE_NORMAL, 0
		);

		// on failure CreateFile returns -1
		if (hFile == (HANDLE)-1)
		{
			MessageBox(NULL, TEXT("Could not open this file"), TEXT("File I/O Error"), MB_ICONSTOP);
			return FALSE;
		}
	}

	return TRUE;
}

LRESULT CALLBACK CKinectApp::OnFileSave(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// here we place the file path 
	TCHAR szFile[MAX_PATH] = { 0 };

	// file extensions filter
	TCHAR *szFilter = TEXT("AVI Files (*.avi)\0*.avi\0All Files\0*.*\0\0");
	//TCHAR *szFilter = TEXT("AVI Files (*.avi)\0*.avi\0MPEG Files (*.mpg)\0*.mpg\0All Files\0*.*\0\0");

	// query the current folder
	TCHAR szCurDir[MAX_PATH];
	::GetCurrentDirectory(MAX_PATH - 1, szCurDir);

	// structure used by the standard file dialog
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(OPENFILENAME));

	// dialog parameters
	ofn.lStructSize	= sizeof(OPENFILENAME);

	// window which owns the dialog
	ofn.hwndOwner = GetParent(hWnd);

	ofn.lpstrFilter	= szFilter;
	// the filters string index (begins with 1)
	ofn.nFilterIndex = 1;
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrDefExt = _T("avi");
	// dialog caption
	ofn.lpstrTitle	= _T("Save");
	ofn.nMaxFileTitle = sizeof(ofn.lpstrTitle);

	ofn.lpfnHook = HookProcCenterDialog((HWND)ofn.hwndOwner, ID_FILE_SAVE, 0, 0);

	// dialog style 
	ofn.Flags = OFN_ENABLEHOOK | OFN_EXPLORER;

	// create and open the dialog (retuns 0 on failure)
	if (GetSaveFileName(&ofn))
	{
		if (_tcslen(ofn.lpstrFile) == 0) return FALSE;
/*
		// try to save the file (which must exist)
		HANDLE hFile = CreateFile(
			ofn.lpstrFile, GENERIC_WRITE,
			FILE_SHARE_WRITE, 0, CREATE_NEW,
			FILE_ATTRIBUTE_NORMAL, 0
		);

		// on failure CreateFile returns -1
		if (hFile == (HANDLE)-1)
		{
			MessageBox(NULL, TEXT("Could not save this file"), TEXT("File I/O Error"), MB_ICONSTOP);
			return FALSE;
		}
*/
		_tcscpy(saveFilePath_, ofn.lpstrFile);
		return TRUE;
	}
	else return FALSE;
}
//--E [] 2012/06/09
