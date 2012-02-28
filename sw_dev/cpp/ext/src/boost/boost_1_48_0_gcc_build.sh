#!/bin/sh

# bjam {stage/install/clean} release debug --toolset=gcc optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete

# bjam stage --toolset=gcc link=static,shared --with-thread --with-regex --with-python
# bjam stage --toolset=gcc link=static,shared --without-python --without-mpi
# bjam stage -sICU_PATH=${ICU_ROOT} --toolset=gcc link=static,shared

./bjam stage release --toolset=gcc --without-mpi link=static,shared
./bjam stage debug --toolset=gcc --without-mpi link=static,shared

# sudo ./bjam install --prefix=/bin/local --toolset=gcc --without-mpi
# ./bjam clean release debug

# ${BOOST_ROOT}/tools/build/v2/user-config.jam

# # -------------------
# # GCC configuration.
# # -------------------

# # Configure specific gcc version (searched for in standard locations and PATH).
# # using gcc : 4.6.1 ;
# # using gcc : 4.6.1 : [c++-compile-command] : [compiler options] ;

# # ---------------------
# # Python configuration.
# # ---------------------

# using python : 3.1 : /usr/bin/ : /usr/include/python2.7 : /usr/lib/python2.7 ;

# using mpi ;
