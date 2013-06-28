#include "spectral.h"
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
//--S [] 2013/06/25: Sang-Wook Lee
//#include <chrono>
#include <boost/chrono.hpp>
#include <boost/foreach.hpp>
//--E [] 2013/06/25: Sang-Wook Lee


#define forsn(i, s, n) for(i = (s); i < (n); ++i)
#define forn(i, n) forsn(i, 0, n)
#define double_infty (numeric_limits<double>::infinity())

using namespace std;

typedef unsigned int uint;
typedef vector<double> mean;
typedef vector<mean> vmean;

template<typename T> double euclidean_distance(const vector<T>& point, const vector<double>& mean) {
    double sum = 0, diff, dim = point.size();
    int i;
    forn(i, dim) {
        diff = point[i] - mean[i];
        sum += diff * diff;
    }
    return sum;
}

void split_largest_cluster(uint e, vcluster& v, vector<int>& count) {
    uint mszi, i, msz = 0, n = v.size(), k = count.size();
    forn(i, k) if(count[i] > msz) {
        msz = count[i];
        mszi = i;
    }
    forn(i, n) {
        if(v[i] != mszi) continue;
        v[i] = e;
        --count[mszi];
        if(++count[e] >= mszi/2) break;
    }
}

template<typename T> double calculate_means(vector<int>& belongs_to, vmean& means, const vector<vector<T>>& points) {
    uint i, j, current_mean, n = belongs_to.size(), k = means.size();
    double res = 0;
    vector<int> point_count(k, 0);
    auto oldmeans = means;
	//--S [] 2013/06/25: Sang-Wook Lee
    //forn(i, k) for(double& coord : means[i]) coord = 0.0;
	forn(i, k) BOOST_FOREACH(double coord, means[i]) { coord = 0.0; }
	//--E [] 2013/06/25: Sang-Wook Lee
    forn(i, n) {
        current_mean = belongs_to[i];
        point_count[current_mean]++;
        forn(j, means[current_mean].size()) means[current_mean][j] += points[i][j];
    }
    forn(i, k) {
        if(point_count[i] == 0) {
            // TODO: Make split_largest_cluster smart enough for this case.
            return -1; // If we lost a cluster, let's try to get lucky next time.
		//--S [] 2013/06/25: Sang-Wook Lee
        //} else for(double& coord : means[i]) coord /= point_count[i];
		} else BOOST_FOREACH(double coord, means[i]) { coord /= point_count[i]; }
		//--E [] 2013/06/25: Sang-Wook Lee
        res += euclidean_distance(means[i], oldmeans[i]);
    }
    return res;
}

template<typename T> void assign_means(vector<int>& belongs_to, const vmean& means, const vector<vector<T>>& points) {
    int i, j, n = points.size(), k = means.size(), disti;
    double dist, newdist;
    forn(i, n) {
        dist = double_infty;
        disti = -1;
        forn(j, k) {
            newdist = euclidean_distance(points[i], means[j]);
            if(newdist >= dist) continue;
            disti = j;
            dist = newdist;
        }
        belongs_to[i] = disti;
    }
}

template<typename T> void initialize(vector<int>& belongs_to, vmean& means, const vector<vector<T>>& points, int k) {
    int n = points.size(), dim = points[0].size();
    belongs_to.clear();
    belongs_to.resize(n, 0);
    means.clear();
    means.resize(k, vector<double>(dim, 0));

    mt19937 gen(static_cast<uint>(time(0) + clock()));
    uniform_int_distribution<uint> partitions(0, k-1);
	//--S [] 2013/06/25: Sang-Wook Lee
    //for(int& x : belongs_to) x = partitions(gen);
	BOOST_FOREACH(int x, belongs_to) { x = partitions(gen); }
	//--E [] 2013/06/25: Sang-Wook Lee
    calculate_means(belongs_to, means, points);
}

template<typename T> vector<int> k_means(const vector<vector<T>>& points, int k, double threshold, uint retries) {
    vector<int> belongs_to; // belongs_to[i] = j means i belongs to cluster j
    vector<vector<double>> means;
    double dist;
    uint tries = 0;
    while(tries++ < retries) {
        initialize(belongs_to, means, points, k);
        do {
            assign_means(belongs_to, means, points);
            dist = calculate_means(belongs_to, means, points);
            if(dist == -1) break;
        } while(dist > threshold);
        if(dist == -1) continue;
        return belongs_to;
    }
    
    cerr << "k-means failed to create " << k << " clusters after " << tries-1 << " tries." << endl;
    // TODO: Handle error. This should not happen.
    return belongs_to;
}