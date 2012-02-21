# bjam --toolset=msvc-8.0 debug release optimization={off/full/space/speed} link=static,shared threading=single,multi runtime-link=single,shared --build-type=complete {stage/install/clean}
# bjam --toolset=msvc link=static,shared --with-thread --with-regex --with-python stage
# bjam --toolset=msvc link=static,shared --without-python --without-mpi stage
# bjam -sICU_PATH=${ICU_ROOT} --toolset=msvc link=static,shared stage

bjam --toolset=gcc debug release link=static,shared --build-type=complete --optimization=full stage

# ${BOOST_ROOT}/tools/build/v2/user-config.jam

# # -------------------
# # GCC configuration.
# # -------------------

# # Configure specific gcc version (searched for in standard locations and PATH).
# # using gcc : 4.6.1 : [c++-compile-command] : [compiler options] ;

# # ---------------------
# # Python configuration.
# # ---------------------

# using python : 3.1 : /usr/bin/ : /usr/include/python2.7 : /usr/lib/python2.7 ;

# using mpi ;
