//------------------------------------------------------------------------------
// <copyright file="DrawDevice.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

// Manages the drawing of bitmap data

#include "stdafx.h"
#include "DrawDevice.h"
#include "KinectApp.h"


inline LONG Width(const RECT &r)
{
    return r.right - r.left;
}

inline LONG Height(const RECT &r)
{
    return r.bottom - r.top;
}

//-------------------------------------------------------------------
// Constructor
//-------------------------------------------------------------------
DrawDevice::DrawDevice()
:   m_hwnd(0),
    m_sourceWidth(0),
    m_sourceHeight(0),
    m_stride(0),
    m_pD2DFactory(NULL), 
    m_pRenderTarget(NULL),
    m_pBitmap(0)
{
}

//-------------------------------------------------------------------
// Destructor
//-------------------------------------------------------------------
DrawDevice::~DrawDevice()
{
    DiscardResources();
    SafeRelease(m_pD2DFactory);
}

//-------------------------------------------------------------------
// EnsureResources
//
// Ensure necessary Direct2d resources are created
//-------------------------------------------------------------------
HRESULT DrawDevice::EnsureResources()
{
    HRESULT hr = S_OK;

    if (!m_pRenderTarget)
    {
        D2D1_SIZE_U size = D2D1::SizeU(m_sourceWidth, m_sourceHeight);

        D2D1_RENDER_TARGET_PROPERTIES rtProps = D2D1::RenderTargetProperties();
        rtProps.pixelFormat = D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE);
        rtProps.usage = D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE;

        // Create a Hwnd render target, in order to render to the window set in initialize
        hr = m_pD2DFactory->CreateHwndRenderTarget(
            rtProps,
            D2D1::HwndRenderTargetProperties(m_hwnd, size),
            &m_pRenderTarget
           );

        if (FAILED(hr))
        {
            return hr;
        }

        // Create a bitmap that we can copy image data into and then render to the target
        hr = m_pRenderTarget->CreateBitmap(
            D2D1::SizeU(m_sourceWidth, m_sourceHeight), 
            D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE)),
            &m_pBitmap 
		);

        if (FAILED(hr))
        {
            SafeRelease(m_pRenderTarget);
            return hr;
        }
    }

    return hr;
}

//-------------------------------------------------------------------
// DiscardResources
//
// Dispose Direct2d resources 
//-------------------------------------------------------------------
void DrawDevice::DiscardResources()
{
    SafeRelease(m_pRenderTarget);
    SafeRelease(m_pBitmap);
}

//-------------------------------------------------------------------
// Initialize
//
// Set the window to draw to, video format, etc.
//-------------------------------------------------------------------
bool DrawDevice::Initialize(HWND hwnd, ID2D1Factory *pD2DFactory, int sourceWidth, int sourceHeight, int Stride)
{
    m_hwnd = hwnd;

    // One factory for the entire application so save a pointer here
    m_pD2DFactory = pD2DFactory;

    m_pD2DFactory->AddRef();

    // Get the frame size
    m_stride = Stride;

    m_sourceWidth = sourceWidth;
    m_sourceHeight = sourceHeight;
    
    return true;
}

//-------------------------------------------------------------------
// DrawFrame
//
// Draw the video frame.
//-------------------------------------------------------------------
bool DrawDevice::Draw(BYTE *pBits, unsigned long cbBits)
{
    // incorrectly sized image data passed in
    if (cbBits < ((m_sourceHeight - 1) * m_stride) + (m_sourceWidth * 4))
    {
        return false;
    }

    // create the resources for this draw device
    // they will be recreated if previously lost
    HRESULT hr = EnsureResources();

    if (FAILED(hr))
    {
        return false;
    }
    
    // Copy the image that was passed in into the direct2d bitmap
    hr = m_pBitmap->CopyFromMemory(NULL, pBits, m_stride);

    if (FAILED(hr))
    {
        return false;
    }
       
    m_pRenderTarget->BeginDraw();

    // Draw the bitmap stretched to the size of the window
    m_pRenderTarget->DrawBitmap(m_pBitmap);
            
    hr = m_pRenderTarget->EndDraw();

    // Device lost, need to recreate the render target
    // We'll dispose it now and retry drawing
    if (hr == D2DERR_RECREATE_TARGET)
    {
        hr = S_OK;
        DiscardResources();
    }

    return SUCCEEDED(hr);
}