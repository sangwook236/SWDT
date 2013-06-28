#include <vector>
#include <functional>
#include <Eigen/Dense>

#define forsn(i, s, n) for(i = (s); i < (n); ++i)
#define forn(i, n) forsn(i, 0, n)
#define double_infty (numeric_limits<double>::infinity())
#define int_infty (numeric_limits<int>::infinity())

using namespace std;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::SelfAdjointEigenSolver;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using Eigen::NoChange;

typedef unsigned int uint;
typedef pair<uint, uint> point;
typedef vector<point> vpoint;
typedef vector<int> vcluster;
typedef function<double(const point&, const point&)> similarity_function;
typedef Matrix<double, Dynamic, Dynamic> matrix;
typedef pair<matrix, matrix> decomposition;

vcluster spectral(const vpoint& points, uint k, double threshold, uint neighbors, double sigma2, uint retries);
vcluster just_k_means(const vpoint& points, uint k, double threshold, uint retries);
