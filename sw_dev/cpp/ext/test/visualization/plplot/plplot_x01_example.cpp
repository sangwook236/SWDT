//#include "stdafx.h"
#include "plc++demos.h"
#include <plevent.h>
#include <boost/smart_ptr.hpp>

#include <cctype>

#ifdef PL_HAVE_UNISTD_H
#include <unistd.h>
#endif

/*
#ifdef PL_USE_NAMESPACE
using namespace std;
#endif
*/

namespace {
namespace local {

static PLGraphicsIn gin;

static int          locate_mode;
static int          test_xor;
static int          fontset;
static char         *f_name;

static const char   *notes[] = { "Make sure you get it right!", "" };

// Options data structure definition.
static PLOptionTable options[] = {
    {
        "locate",               // Turns on test of API locate function.
        NULL,
        NULL,
        &locate_mode,
        PL_OPT_BOOL,
        "-locate",
        "Turns on test of API locate function"
    },
    {
        "xor",                  // Turns on test of xor function.
        NULL,
        NULL,
        &test_xor,
        PL_OPT_BOOL,
        "-xor",
        "Turns on test of XOR"
    },
    {
        "font",                 // For switching between font set 1 & 2.
        NULL,
        NULL,
        &fontset,
        PL_OPT_INT,
        "-font number",
        "Selects stroke font set (0 or 1, def:1)"
    },
    {
        "save",                 // For saving in postscript.
        NULL,
        NULL,
        &f_name,
        PL_OPT_STRING,
        "-save filename",
        "Save plot in color postscript 'file'"
    },
    {
        NULL,                   // Option.
        NULL,                   // Handler.
        NULL,                   // Client data.
        NULL,                   // Address of variable to set.
        0,                      // Mode flag.
        NULL,                   // Short syntax.
        NULL
    }                           // Long syntax.
};


class x01
{
public:
    x01(int argc, char *argv[]);

    void plot1(int);
    void plot2();
    void plot3();


private:
    // Class data
    PLFLT    xscale, yscale, xoff, yoff;
    plstream *pls;
};


x01::x01(int argc, char *argv[])
{
    char ver[80];

    fontset = 1;
    f_name  = NULL;

    // plplot initialization.
    // Divide page into 2x2 plots unless user overrides.

    pls = new plstream();

    // Parse and process command line arguments.
    pls->MergeOpts( options, "x01c options", notes );
#if defined(_WIN64) || defined(_WIN32)
	pls->parseopts( &argc, (char**)argv, PL_PARSE_FULL );
#else
	pls->parseopts(&argc, (const char**)argv, PL_PARSE_FULL);
#endif

    // Print out version number, just for kicks.
    pls->gver( ver );

    std::cout << "PLplot library version: " << ver << std::endl;


    // Initialize PLplot.
    pls->star( 2, 2 );

    // Select the multi-stroke font.
    pls->fontld( 1 );

    // Set up the data.
    // Original case.

    xscale = 6.;
    yscale = 1.;
    xoff   = 0.;
    yoff   = 0.;

    // Do a plot.
    plot1( 0 );

    // Set up the data.

    xscale = 1.;
    yscale = 0.0014;
    yoff   = 0.0185;

    // Do a plot.

    int digmax = 5;
    pls->syax( digmax, 0 );

    plot1( 1 );

    plot2();

    plot3();

    // Show how to save a plot:
    // Open a new device, make it current, copy parameters, and replay the plot buffer.

    if (f_name) // command line option '-save filename'.
    {
        std::cout << "The current plot was saved in color Postscript under the name '" << f_name << "'" << std::endl;
        plstream *pls2;

        pls2 = new plstream();       // Ceate a new one.

        pls2->sfnam( f_name );       // File name.
        pls2->sdev( "psc" );         // Device type.

        pls2->cpstrm( *pls, false ); // Dopy old stream parameters to new stream.
        pls2->replot();              // Do the save by replaying the plot buffer.

        delete pls2;
    }

    // Let's get some user input.

    if ( locate_mode )
    {
        while ( 1 )
        {
            if ( !pls->GetCursor( &gin ) )
                break;
            if ( gin.keysym == PLK_Escape )
                break;

            pls->text();
            if ( gin.keysym < 0xFF && isprint( gin.keysym ) )
                std::cout << "wx = " << gin.wX << ", wy = " << gin.wY <<
                ", dx = " << gin.dX << ",  dy = " << gin.dY <<
                ",  c = '" << gin.keysym << "'" << std::endl;
            else
                std::cout << "wx = " << gin.wX << ", wy = " << gin.wY <<
                ", dx = " << gin.dX << ",  dy = " << gin.dY <<
                ",  c = '" << gin.keysym << "'" << std::endl;
            //	printf("wx = %f,  wy = %f, dx = %f,  dy = %f,  c = 0x%02x\n",
            //                              gin.wX, gin.wY, gin.dX, gin.dY, gin.keysym);

            pls->gra();
        }
    }

    // In C++ we don't call plend() to finish off.
    // This is handled by the destructor.
    delete pls;
}

void x01::plot1( int do_test )
{
    int   i;
    PLFLT xmin, xmax, ymin, ymax;
    PLFLT *x  = new PLFLT[60];
    PLFLT *y  = new PLFLT[60];
    PLFLT *xs = new PLFLT[6];
    PLFLT *ys = new PLFLT[6];

    for ( i = 0; i < 60; i++ )
    {
        x[i] = xoff + xscale * ( i + 1 ) / 60.0;
        y[i] = yoff + yscale * pow( (double) x[i], 2. );
    }

    xmin = x[0];
    xmax = x[59];
    ymin = y[0];
    ymax = y[59];

    for ( i = 0; i < 6; i++ )
    {
        xs[i] = x[i * 10 + 3];
        ys[i] = y[i * 10 + 3];
    }

    // Set up the viewport and window using PLENV. The range in X is 0.0 to
    // 6.0, and the range in Y is 0.0 to 30.0. The axes are scaled separately
    // (just = 0), and we just draw a labelled box (axis = 0).

    pls->col0( 1 );
    pls->env( xmin, xmax, ymin, ymax, 0, 0 );
    pls->col0( 2 );
    pls->lab( "(x)", "(y)", "#frPLplot Example 1 - y=x#u2" );

    // Plot the data points.

    pls->col0( 4 );
    pls->poin( 6, xs, ys, 9 );

    // Draw the line through the data.

    pls->col0( 3 );
    pls->line( 60, x, y );

// xor mode enable erasing a line/point/text by replotting it again
// it does not work in double buffering mode, however.

    if ( do_test && test_xor )
    {
#ifdef PL_HAVE_USLEEP
        bool st;
        pls->xormod( true, &st ); // Enter xor mode.
        if ( st )
        {
            for ( i = 0; i < 60; i++ )
            {
                pls->poin( 1, x + i, y + i, 9 );   // Draw a point.
                usleep( 50000 );                   // Wait a little.
                pls->flush();                      // Force an update of the tk driver.
                pls->poin( 1, x + i, y + i, 9 );   // Erase point.
            }
            pls->xormod( false, &st );             // Leave xor mode.
        }
#else
        std::cout << "The -xor command line option can only be exercised if your system has usleep(), which does not seems to happen." << std::endl;
#endif
    }

    delete[] x;
    delete[] y;
    delete[] xs;
    delete[] ys;
}

void x01::plot2()
{
    int   i;
    PLFLT *x = new PLFLT[100];
    PLFLT *y = new PLFLT[100];

    // Set up the viewport and window using PLENV. The range in X is -2.0 to
    // 10.0, and the range in Y is -0.4 to 2.0. The axes are scaled
    // separately (just = 0), and we draw a box with axes (axis = 1).

    pls->col0( 1 );
    pls->env( -2.0, 10.0, -0.4, 1.2, 0, 1 );
    pls->col0( 2 );
    pls->lab( "(x)", "sin(x)/x", "#frPLplot Example 1 - Sinc Function" );

    // Fill up the arrays.

    for ( i = 0; i < 100; i++ )
    {
        x[i] = ( i - 19.0 ) / 6.0;
        y[i] = 1.0;
        if ( x[i] != 0.0 )
            y[i] = sin( x[i] ) / x[i];
    }

    // Draw the line.

    pls->col0( 3 );
    pls->width( 2 );
    pls->line( 100, x, y );
    pls->width( 1 );

    delete[] x;
    delete[] y;
}

void x01::plot3()
{
    int   i;
    PLINT space0 = 0;
    PLINT mark0  = 0;
    PLINT space1 = 1500;
    PLINT mark1  = 1500;
    PLFLT *x     = new PLFLT[101];
    PLFLT *y     = new PLFLT[101];

    // For the final graph we wish to override the default tick intervals, and so do not use plenv().

    pls->adv( 0 );

    // Use standard viewport, and define X range from 0 to 360 degrees, Y range from -1.2 to 1.2.

    pls->vsta();
    pls->wind( 0.0, 360.0, -1.2, 1.2 );

    // Draw a box with ticks spaced 60 degrees apart in X, and 0.2 in Y.

    pls->col0( 1 );
    pls->box( "bcnst", 60.0, 2, "bcnstv", 0.2, 2 );

    // Superimpose a dashed line grid, with 1.5 mm marks and spaces.
    // plstyl expects a pointer!

    pls->styl( 1, &mark1, &space1 );
    pls->col0( 2 );
    pls->box( "g", 30.0, 0, "g", 0.2, 0 );
    pls->styl( 0, &mark0, &space0 );

    pls->col0( 3 );
    pls->lab( "Angle (degrees)", "sine", "#frPLplot Example 1 - Sine function" );

    for ( i = 0; i < 101; i++ )
    {
        x[i] = 3.6 * i;
        y[i] = sin( x[i] * M_PI / 180.0 );
    }

    pls->col0( 4 );
    pls->line( 101, x, y );

    delete[] x;
    delete[] y;
}

}  // local
}  // unnamed namespace

namespace my_plplot {

void x01_example(const int argc, char *argv[])
{
    boost::scoped_ptr<local::x01> x(new local::x01(argc, argv));
}

}  // namespace my_plplot
