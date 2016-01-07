//#include "stdafx.h"
#include "AdisUsbz.h"
#include <assert.h>
#include <windows.h>


#define IOCTL_BULK_WRITE ( 0x220001 + ( 0x800 + 20 ) * 4 )
#define IOCTL_BULK_READ  ( 0x220002 + ( 0x800 + 19 ) * 4 )

AdisUsbz::AdisUsbz() : mHandle( INVALID_HANDLE_VALUE )
{
}

AdisUsbz::~AdisUsbz()
{
    // If the instance have been initialized, we need to close the handle to
    // the driver.
    if ( mHandle != INVALID_HANDLE_VALUE )
    {
        CloseHandle( mHandle );
        mHandle = INVALID_HANDLE_VALUE;
    }
}

#if defined(UNICODE) || defined(_UNICODE)
bool AdisUsbz::Initialize( const wchar_t * aDeviceName )
#else
bool AdisUsbz::Initialize( const char * aDeviceName )
#endif
{
    assert( aDeviceName != NULL );

    assert( mHandle == INVALID_HANDLE_VALUE );

    mHandle = CreateFile( aDeviceName, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL );
    if ( mHandle == INVALID_HANDLE_VALUE )
    {
        return false;
    }

    WriteStaticBits();
    WriteLoopDelay ();

    return true;
}

short AdisUsbz::ReadInt14( unsigned char aAddress )
{
    short lResult = ReadInt16( aAddress );

    // Sign extension from 14 bits to 16 bits
    if ( ( lResult & 0x2000 ) == 0x2000 )
    {
        lResult |= 0xc000;
    }
    else
    {
        lResult &= 0x3fff;
    }

    return lResult;
}

// Private
/////////////////////////////////////////////////////////////////////////////

short AdisUsbz::ReadInt16( unsigned char aAddress )
{
	unsigned char lBuffer[ 7 ];

    memset( lBuffer, 0, sizeof( lBuffer ) );

    lBuffer[ 0 ] = 0x0d; // Polarity   = POS
    lBuffer[ 1 ] = 0x01; // Channel    = 1
    lBuffer[ 2 ] = 2;    // Byte count = 2
    lBuffer[ 3 ] = aAddress;
    lBuffer[ 5 ] = aAddress;

    Tx( lBuffer, 7 );
    Rx( lBuffer, 4 );

    // Reverse byte order
    return ( ( lBuffer[ 2 ] << 8 ) | lBuffer[ 3 ] );
}

void AdisUsbz::WriteLoopDelay()
{
    unsigned char lBuffer[ 2 ];

    lBuffer[ 0 ] = 0x0b;
    lBuffer[ 1 ] = 25;

    Tx( lBuffer, 2 );
}

void AdisUsbz::WriteStaticBits()
{
    unsigned char lBuffer[ 2 ];

    lBuffer[ 0 ] = 0x0c;
    lBuffer[ 1 ] = 0x71;

    Tx( lBuffer, 2 );
}

void AdisUsbz::Rx( void * aOut, unsigned char aOutSize )
{
    assert( aOut    != NULL );
    assert( aOutSize > 0    );

    unsigned int lBTCT = 1; // Bulk Transfer Control Type

    DeviceControl( IOCTL_BULK_WRITE, & lBTCT, sizeof( lBTCT ), aOut, aOutSize );
}

void AdisUsbz::Tx( const void * aIn, unsigned char aInSize )
{
    assert( aIn    != NULL );
    assert( aInSize > 0    );

    unsigned int  lBTCT = 0; // Bulk Transfer Control Type

    DeviceControl( IOCTL_BULK_WRITE, & lBTCT, sizeof( lBTCT ), const_cast< void * >( aIn ), aInSize );
}

void AdisUsbz::DeviceControl( unsigned int aIoCtl, const void * aIn, unsigned int aInSize, void * aOut, unsigned int aOutSize )
{
    assert( mHandle != INVALID_HANDLE_VALUE );

    DWORD lInfo;

    BOOL lBool = DeviceIoControl( mHandle, aIoCtl, const_cast< void * >( aIn ), aInSize, aOut, aOutSize, & lInfo, NULL );
    assert( lBool != FALSE    );
    assert( lInfo == aOutSize );
}
