#include <iostream>
#include <vector>
#include <cmath>

#define forsn(i, s, n) for(i = (s); i < (n); ++i)
#define forn(i, n) forsn(i, 0, n)

using namespace std;

typedef unsigned int uint;
typedef unsigned char uchar;

void hsb_to_rgb(double h, double s, double v, double& r, double& g, double& b) {
    if (s == 0.0) { 
        r = g = b = v;
        return;
    }
    double f, p, q, t;
    int i;

    if (h == 360.0) h = 0.0;
    h /= 60.0;
    i = (int) floor(h);
    f = h - i;
    p = v * (1.0 - s);
    q = v * (1.0 - (s*f));
    t = v * (1.0 - (s*(1.0-f)));
    switch(i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
    }
}

int main() {
    int i = 0, n, k;
    double hue, r, g, b;
    cin >> n >> k;
    vector<int> clusters(n);
    vector<pair<double, double>> points(n, make_pair(0, 0));
    
    forn(i, n) cin >> clusters[i];
    forn(i, n) cin >> points[i].first >> points[i].second;
    
    cout << "import numpy as np" << endl;
    cout << "import matplotlib.pyplot as plt" << endl << endl;
    cout << "x = [";
    forn(i, n-1) cout << points[i].first << ", ";
    cout << points[n-1].first << "]" << endl;
    
    cout << "y = [";
    forn(i, n-1) cout << points[i].second << ", ";
    cout << points[n-1].second << "]" << endl;
    
    cout << "c = [";
    forn(i, n) {
        hue = ((double)(clusters[i]+1))/((double)k);
        hsb_to_rgb(360.0f*hue, 0.62, 0.69, r, g, b);
        cout << "[" << r << ", " << g << ", " << b << "]";
        if(i < n-1) cout << ", ";
    }
    cout << "]" << endl;
    
    cout << "plt.scatter(x, y, 80, c)" << endl;
    cout << "plt.show()" << endl;
}