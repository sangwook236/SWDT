#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include "spectral.h"

using boost::lexical_cast;
using namespace std;
namespace po = boost::program_options;


int main(int argc, char** argv) {
    uint x, y, k, neighbors, n, retries;
    double threshold, sigma2;
    string method;
    // Declare the supported options.
    po::options_description desc("Program options");
    desc.add_options()
        ("help,h", "describe program usage")
        ("clusters,k", po::value<uint>(&k)->default_value(1), "number of clusters to attempt")
        ("neighbors,e", po::value<uint>(&neighbors)->default_value(1), "number of neighbors to consider in the graph")
        ("sigma2,s", po::value<double>(&sigma2)->default_value(10000), "fallout speed of gaussian metric")
        ("threshold,t", po::value<double>(&threshold)->default_value(0.01), "k-means clustering threshold")
        ("method", po::value<string>(&method)->default_value("spectral"), "method, either spectral or kmeans")
        ("retries", po::value<uint>(&retries)->default_value(50000), "number of k-means attempts at making k clusters before giving up")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help") || (method != "spectral" && method != "kmeans")) {
        cout << desc << "\n";
        return 1;
    }
    
    vpoint points;
    while(cin >> x >> y) points.push_back(point(x,y));
    n = points.size();
    
    auto clusters = (method == "spectral") ? spectral(points, k, threshold, neighbors, sigma2, retries)
                                           : just_k_means(points, k, threshold, retries);
    
    cout << n << endl;
    cout << k << endl;
    for(auto x: clusters) cout << x << " ";
    cout << endl;
    for(auto x: points) cout << x.first << " " << x.second << endl;
}
