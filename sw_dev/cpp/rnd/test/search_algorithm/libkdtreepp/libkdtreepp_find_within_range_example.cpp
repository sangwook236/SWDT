//#include "stdafx.h"
#include <kdtree++/kdtree.hpp>
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <iterator>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libkdtreepp {

struct kdtreeNode
{
    typedef double value_type;

    double xyz[3];
    size_t index;

    value_type operator[](size_t n) const
    {
        return xyz[n];
    }

    double distance(const kdtreeNode &node)
    {
        double x = xyz[0] - node.xyz[0];
        double y = xyz[1] - node.xyz[1];
        double z = xyz[2] - node.xyz[2];

        // this is not correct   return sqrt( x*x+y*y+z*z);

        // this is what kdtree checks with find_within_range()
        // the "manhattan distance" from the search point.
        // effectively, distance is the maximum distance in any one dimension.
        return std::max(std::fabs(x), std::max(std::fabs(y), std::fabs(z)));
    }
};

void find_within_range_example()
{
    std::vector<kdtreeNode> pts;

    typedef KDTree::KDTree<3, kdtreeNode> treeType;

    treeType tree;

    // make random 3d points
    for (size_t n = 0; n < 10000; ++n)
    {
        kdtreeNode node;
        node.xyz[0] = double(std::rand()) / RAND_MAX;
        node.xyz[1] = double(std::rand()) / RAND_MAX;
        node.xyz[2] = double(std::rand()) / RAND_MAX;
        node.index = n;

        tree.insert(node);
        pts.push_back(node);
    }

    for (size_t r = 0; r < 1000; ++r)
    {
        kdtreeNode refNode;
        refNode.xyz[0] = double(std::rand()) / RAND_MAX;
        refNode.xyz[1] = double(std::rand()) / RAND_MAX;
        refNode.xyz[2] = double(std::rand()) / RAND_MAX;

        double limit = double(std::rand()) / RAND_MAX;

        // find the correct return list by checking every single point
        std::set<size_t> correctCloseList;

        for (size_t i= 0; i < pts.size(); ++i)
        {
            const double dist = refNode.distance(pts[i]);
            if (dist < limit)
                correctCloseList.insert(i);
        }

        // now do the same with the kdtree.
        std::vector<kdtreeNode> howClose;
        tree.find_within_range(refNode, limit, std::back_insert_iterator<std::vector<kdtreeNode> >(howClose));

        // make sure no extra points are returned, and the return has no missing points.
        for (size_t i = 0; i < howClose.size(); ++i)
        {
            std::set<size_t>::iterator hit = correctCloseList.find(howClose[i].index);

            if (hit != correctCloseList.end())
            {
                correctCloseList.erase(hit);
            }
            else
            {
                // point that is too far away - fail!
                assert(false);
                std::cerr << "fail, extra points." << std::endl;
            }
        }

        // fail, not all of the close enough points returned.
        assert(correctCloseList.size() == 0);
        if (correctCloseList.size() > 0)
        {
            std::cerr << "fail, missing points." << std::endl;
        }
    }
}

}  // namespace my_libkdtreepp

