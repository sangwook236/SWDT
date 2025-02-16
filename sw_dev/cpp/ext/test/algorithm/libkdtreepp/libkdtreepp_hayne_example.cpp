//#include "stdafx.h"
#define KDTREE_SIZE_T unsigned int
#include <kdtree++/kdtree.hpp>
#include <iostream>
#include <functional>
#include <set>
#include <vector>
#include <limits>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libkdtreepp {

struct duplet
{
    typedef int value_type;

    inline value_type operator[](int const N) const
    {
         return d[N];
    }

    inline bool operator==(duplet const &other) const
    {
        return this->d[0] == other.d[0] && this->d[1] == other.d[1];
    }

    inline bool operator!=(duplet const &other) const
    {
        return this->d[0] != other.d[0] || this->d[1] != other.d[1];
    }

    friend std::ostream & operator<<(std::ostream &o, duplet const &d)
    {
        return o << "(" << d[0] << "," << d[1] << ")";
    }

    value_type d[2];
};

typedef KDTree::KDTree<2, duplet, std::pointer_to_binary_function<duplet, int, double> > duplet_tree_type;

inline double return_dup(duplet d, int k)
{
     return d[k];
}

// ${LIBKDTREE++_HOME}/examples/test_hayne.cpp.
void hayne_example()
{
    duplet_tree_type dupl_tree_test(std::ptr_fun(return_dup));
    std::vector<duplet> vDuplets;

    //srand(time(0));
    int randy1 = 0;
    int randy2 = 0;
    for (int i = 0; i < 700; ++i)
    {
        // create coordinate for new duplet.
        randy1 += 2;
        randy1 = randy1 % 255;
        randy2 += 3;
        randy2 = randy2 % 255;
        //randy1 = rand() % 255;
        //randy2 = rand() % 255;

        // new duplet.
        duplet super_dupre = { { randy1, randy2 } };

        // check if duplet with same coordinate already in vector/tree. If not: insert in vector and tree.
        duplet_tree_type::iterator pItr = dupl_tree_test.find_nearest(super_dupre, std::numeric_limits<double>::max()).first;
        if (*pItr != super_dupre)
        {
            dupl_tree_test.insert(super_dupre);
            vDuplets.push_back(super_dupre);
        }
    }

    dupl_tree_test.optimise();

    while (vDuplets.size() > 0)  // delete all duplets from tree which are in the vector.
    {
        size_t elements = vDuplets.size();

        duplet element_to_erase = vDuplets.back();
        vDuplets.pop_back();

        if (vDuplets.size() == 147)
            std::cout << "THIS IS THE BUG TRIGGER" << std::endl;

        std::cout << vDuplets.size() << " : Deleting " << element_to_erase << std::endl;

        assert(std::find(dupl_tree_test.begin(),dupl_tree_test.end(), element_to_erase) != dupl_tree_test.end());
        assert(dupl_tree_test.find(element_to_erase) != dupl_tree_test.end());

        duplet_tree_type::iterator will = dupl_tree_test.find(element_to_erase);
        duplet_tree_type::iterator should = dupl_tree_test.find_exact(element_to_erase);

        std::cout << "  tree will delete:   " << *will << std::endl;
        std::cout << "  tree should delete: " << *should << std::endl;

        assert(*will == *should);

        dupl_tree_test.erase(element_to_erase); //erase() : will probably erase wrong element sooner or later
        //dupl_tree_test.erase_exact(element_to_erase); --> this works

        // now check that it cannot find the element UNLESS there is another one with the identical location in the list...
        if (std::find(vDuplets.begin(), vDuplets.end(), element_to_erase) == vDuplets.end())
        {
            duplet_tree_type::iterator not_there = dupl_tree_test.find(element_to_erase);
            if (not_there != dupl_tree_test.end())
            {
                std::cout << "SHOULD NOT HAVE FOUND THIS: " << *not_there << std::endl;
                assert(false);
            }
            else
            {
                std::cout << "  find() double-check passed." << std::endl;
            }
        }
    }
}

}  // namespace my_libkdtreepp
