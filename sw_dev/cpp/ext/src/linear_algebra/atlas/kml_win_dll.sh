#! /bin/bash

GCC=gcc
defname=atlas_kml.def
dllname=atlas_kml.dll
libbat=generate_implib.bat

#--------------------------------------------
# .DEF file generation
#--------------------------------------------
echo 'Generating the .def file...'

Cexports=(\
#single precision routines
	cblas_sdsdot cblas_sdot cblas_snrm2 cblas_sasum cblas_scnrm2 \
	cblas_scasum cblas_sscal cblas_isamax \
	cblas_sswap cblas_scopy cblas_saxpy catlas_saxpby catlas_sset \
	cblas_srotg cblas_srotmg cblas_srot cblas_srotm \
	cblas_sgemv cblas_sgbmv cblas_strmv cblas_stbmv cblas_stpmv \
	cblas_strsv cblas_stbsv cblas_stpsv \
	cblas_ssymv	cblas_ssbmv cblas_sspmv cblas_sger cblas_ssyr \
	cblas_sspr cblas_ssyr2 cblas_sspr2 \
	cblas_sgemm cblas_ssymm cblas_ssyrk cblas_ssyr2k cblas_strmm cblas_strsm \
#double precision routines
	cblas_dsdot cblas_ddot cblas_dnrm2 cblas_dasum cblas_dznrm2 \
	cblas_dzasum cblas_dscal cblas_idamax \
	cblas_dswap cblas_dcopy cblas_daxpy catlas_daxpby catlas_dset \
	cblas_drotg cblas_drotmg cblas_drot cblas_drotm \
	cblas_dgemv cblas_dgbmv cblas_dtrmv cblas_dtbmv cblas_dtpmv \
	cblas_dtrsv cblas_dtbsv cblas_dtpsv \
	cblas_dsymv cblas_dsbmv cblas_dspmv cblas_dger cblas_dsyr \
	cblas_dspr cblas_dsyr2 cblas_dspr2 \
	cblas_dgemm cblas_dsymm cblas_dsyrk cblas_dsyr2k cblas_dtrmm cblas_dtrsm \
#single precision complex routines
#	cblas_cdotu_sub cblas_cdotc_sub cblas_cscal cblas_csscal cblas_icamax \
#	cblas_cswap cblas_ccopy cblas_caxpy catlas_caxpby catlas_cset \
#	cblas_crotg cblas_csrot \
#	cblas_cgemv cblas_cgbmv	cblas_ctrmv cblas_ctbmv cblas_ctpmv \
#	cblas_ctrsv cblas_ctbsv cblas_ctpsv \
#	cblas_chemv cblas_chbmv cblas_chpmv cblas_cgeru cblas_cgerc \
#	cblas_cher cblas_chpr cblas_cher2 cblas_chpr2 \
#	cblas_cgemm cblas_csymm cblas_csyrk cblas_csyr2k cblas_ctrmm cblas_ctrsm \
#	cblas_chemm cblas_cherk cblas_cher2k \
#double precision complex routines
#	cblas_zdotu_sub cblas_zdotc_sub cblas_zscal cblas_zdscal cblas_izamax \
#	cblas_zswap cblas_zcopy cblas_zaxpy catlas_zaxpby catlas_zset \
#	cblas_zrotg cblas_zdrot \
#	cblas_zgemv cblas_zgbmv cblas_ztrmv cblas_ztbmv cblas_ztpmv	\
#	cblas_ztrsv cblas_ztbsv cblas_ztpsv \
#	cblas_zhemv cblas_zhbmv cblas_zhpmv cblas_zgeru cblas_zgerc \
#	cblas_zher cblas_zhpr cblas_zher2 cblas_zhpr2 \
#	cblas_zgemm cblas_zsymm cblas_zsyrk cblas_zsyr2k cblas_ztrmm cblas_ztrsm \
#	cblas_zhemm cblas_zherk cblas_zher2k \
#other stuff
#	cblas_errprn \
# CLAPACK interface subset
	clapack_sgesv clapack_sgetrf clapack_sgetrs clapack_sgetri \
	clapack_sposv clapack_spotrf clapack_spotrs clapack_spotri \
	clapack_slauum clapack_strtri \
	clapack_dgesv clapack_dgetrf clapack_dgetrs clapack_dgetri \
	clapack_dposv clapack_dpotrf clapack_dpotrs clapack_dpotri \
	clapack_dlauum clapack_dtrtri \
	clapack_cgesv clapack_cgetrf clapack_cgetrs clapack_cgetri \
	clapack_cposv clapack_cpotrf clapack_cpotrs clapack_cpotri \
	clapack_clauum clapack_ctrtri \
	clapack_zgesv clapack_zgetrf clapack_zgetrs clapack_zgetri \
	clapack_zposv clapack_zpotrf clapack_zpotrs clapack_zpotri \
	clapack_zlauum clapack_ztrtri)


#--------------------------------------------
# Generate a .def file
#--------------------------------------------
echo "EXPORTS" > ${defname}
for (( i = 0 ; i < ${#Cexports[*]} ; i++ )) ; do
	export=${Cexports[$i]}
	echo -e "${export}=${export}" >> ${defname}
done

#--------------------------------------------
# Making a .DLL from the .a files
#--------------------------------------------
echo 'Generating windows .dll file...'

#-----------------(method 1)
#CLIBPATH=/usr/lib/mingw
#mingwclib="$CLIBPATH/libg2c.a $CLIBPATH/libmoldname.a $CLIBPATH/libmsvcrt.a"
#${GCC} -mno-cygwin -shared -o ${dllname} ${defname} \
#    liblapack.a libcblas.a libf77blas.a libatlas.a \
#    -Wl,--enable-auto-import \
#    -Wl,--no-whole-archive ${mingwclib}
#-----------------(method 2)
GCC_MINGW=i686-pc-mingw32-gcc
CLIBPATH=/usr/i686-pc-mingw32/sys-root/mingw/lib
mingwclib="$CLIBPATH/libmoldname100.a $CLIBPATH/libmsvcrt.a"
${GCC_MINGW} -shared -o ${dllname} ${defname} \
    liblapack.a libcblas.a libf77blas.a libatlas.a \
    -Wl,--enable-auto-import \
    -Wl,--no-whole-archive ${mingwclib}
#-----------------(method 3)
#GCC_MINGW=i686-pc-mingw32-gcc
#CLIBPATH1=/usr/lib/gcc/i686-pc-mingw32/4.5.2/
#CLIBPATH2=/usr/i686-pc-mingw32/sys-root/mingw/lib
#mingwclib="$CLIBPATH2/libmoldname100.a $CLIBPATH2/libmsvcrt.a $CLIBPATH2/libpthread.a $CLIBPATH1/libgfortran.a"
#${GCC_MINGW} -shared -o ${dllname} ${defname} \
#    -Wl,--enable-auto-import \
#    -Wl,--whole-archive liblapack.a libcblas.a libf77blas.a libatlas.a \
#    -Wl,--no-whole-archive ${mingwclib}


#--------------------------------------------
# Making a .bat for the import-library
#
# TODO: figure out a better way to find vcvars32.bat
#--------------------------------------------
echo 'Generating the import library script...'
echo " " > $libbat
echo -e 'call "C:/Program Files (x86)/Microsoft Visual Studio 10.0/Vc/bin/vcvars32.bat"' >> $libbat
#echo -e 'call "C:/Program Files/Microsoft Visual Studio .NET 2003/Vc7/bin/vcvars32.bat"' >> $libbat
#echo -e 'call "C:/Program Files/Microsoft Visual Studio/VC98/bin/vcvars32.bat"' >> $libbat
#echo -e 'call "C:/Program Files/DevStudio/VC/bin/vcvars32.bat"' >> $libbat
#echo -e 'call "C:/Msdev/bin/vcvars32.bat x86"' >> $libbat
echo -e 'lib.exe /DEF:atlas_kml.def /MACHINE:Ix86 /OUT:atlas_kml.lib' >> $libbat

echo 'Running the import library script...'
chmod +x ./$libbat
./$libbat

mkdir win
mv atlas_kml.dll win
mv atlas_kml.lib win
cp ../../include/cblas.h win
cp ../../include/clapack.h win
