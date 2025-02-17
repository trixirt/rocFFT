# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

cmake_minimum_required( VERSION 3.16 )

# This should appear before the project command, because it does not
# use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH
    "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH
    "Install path prefix, prepended onto install directories" )
endif( )



# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user
# specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( rocfft-clients-tests LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if( NOT TARGET rocfft )
  find_package( rocfft REQUIRED CONFIG PATHS )
endif( )

if( NOT HIP_FOUND )
  find_package( HIP REQUIRED )
endif()

if( NOT ROCM_FOUND )
  find_package( ROCM 0.7.3 REQUIRED )
endif()

if( NOT hiprand_FOUND )
  find_package( hiprand REQUIRED )
endif()

include( ROCMInstallTargets )

set( rocfft-test_source
  gtest_main.cpp
  rocfft_accuracy_test.cpp
  accuracy_test.cpp
  accuracy_test_1D.cpp	
  accuracy_test_2D.cpp
  accuracy_test_3D.cpp
  accuracy_test_adhoc.cpp
  accuracy_test_callback.cpp
  accuracy_test_checkstride.cpp
  multithread_test.cpp
  hermitian_test.cpp
  hipGraph_test.cpp
  default_callbacks_test.cpp
  unit_test.cpp
  misc/source/test_exception.cpp
  validate_length_stride.cpp
  random.cpp
  ../../shared/array_validator.cpp
  )

set( rocfft-test_includes
  fftw_transform.h
  rocfft_against_fftw.h
  misc/include/test_exception.h
  )

add_executable( rocfft-test ${rocfft-test_source} ${rocfft-test_includes} )
add_executable( rtc_helper_crash rtc_helper_crash.cpp )

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_USE_STATIC_LIBS OFF )

option( BUILD_FFTW "Download and build FFTW" OFF )

# look for installed FFTW if we weren't asked to build it
if( NOT BUILD_FFTW )
  find_package( FFTW 3.0 MODULE COMPONENTS FLOAT DOUBLE )
endif()

include( ExternalProject )

# also try to build FFTW if FFTW isn't present
if( BUILD_FFTW OR NOT FFTW_FOUND )
  set(FFTW_LIBRARIES_DOUBLE
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3_threads${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_double-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(FFTW_LIBRARIES_SINGLE
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3f_threads${CMAKE_SHARED_LIBRARY_SUFFIX}
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw_single-build/${CMAKE_SHARED_LIBRARY_PREFIX}fftw3f${CMAKE_SHARED_LIBRARY_SUFFIX})

  set(FFTW_CMAKE_ARGS_COMMON
      -DDISABLE_FORTRAN=ON
      -DENABLE_AVX2=ON
      -DENABLE_THREADS=ON
      -DBUILD_SHARED_LIBS=ON
      -DBUILD_TESTS=OFF
      -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER})

  set(FFTW_SRC_URL http://www.fftw.org/fftw-3.3.9.tar.gz CACHE STRING "Location of FFTW source code")
  set(FFTW_SRC_SHA256 bf2c7ce40b04ae811af714deb512510cc2c17b9ab9d6ddcf49fe4487eea7af3d CACHE STRING "SHA256 hash of FFTW source code")

  # build double-precision FFTW
  ExternalProject_Add(fftw_double
                      URL ${FFTW_SRC_URL}
                      URL_HASH SHA256=${FFTW_SRC_SHA256}
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
                      CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON}
                      INSTALL_COMMAND ""
                      BUILD_BYPRODUCTS ${FFTW_LIBRARIES_DOUBLE})
  ExternalProject_Get_Property( fftw_double source_dir binary_dir )

  # also build single-precision fftw from the same source dir
  ExternalProject_Add(fftw_single
                      DOWNLOAD_COMMAND ""
                      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/fftw
                      PREFIX ${CMAKE_CURRENT_BINARY_DIR}
                      CMAKE_ARGS ${FFTW_CMAKE_ARGS_COMMON} -DENABLE_FLOAT=ON
                      INSTALL_COMMAND ""
                      BUILD_BYPRODUCTS ${FFTW_LIBRARIES_SINGLE}
                      DEPENDS fftw_double)
  ExternalProject_Get_Property( fftw_single source_dir binary_dir )

  set(FFTW_INCLUDES
      ${CMAKE_CURRENT_BINARY_DIR}/src/fftw/api)
  set(FFTW_LIBRARIES
      ${FFTW_LIBRARIES_DOUBLE}
      ${FFTW_LIBRARIES_SINGLE})

  # FFTW we build is always threaded
  set( FFTW_MULTITHREAD TRUE )

  add_dependencies( rocfft-test fftw_double fftw_single )
  rocm_install(
    FILES ${FFTW_LIBRARIES}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/fftw
    COMPONENT clients-common
  )
endif()

set( rocfft-test_include_dirs
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${FFTW_INCLUDES}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/misc/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
  ${ROCM_CLANG_ROOT}/include
  )

set( rocfft-test_link_libs
  ${FFTW_LIBRARIES}
  Boost::program_options
  )

include( ../cmake/build-gtest.cmake )

if( BUILD_GTEST OR NOT GTEST_FOUND )
  add_dependencies( rocfft-test gtest )
  list( APPEND rocfft-test_include_dirs ${GTEST_INCLUDE_DIRS} )
  list( APPEND rocfft-test_link_libs ${GTEST_LIBRARIES} )
else()  
  list( APPEND rocfft-test_include_dirs $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}> )
  list( APPEND rocfft-test_link_libs ${GTEST_LIBRARIES} )
endif()

target_compile_options( rocfft-test PRIVATE ${WARNING_FLAGS} -Wno-cpp )

if( ROCFFT_RUNTIME_COMPILE )
  target_compile_options( rocfft-test PRIVATE -DROCFFT_RUNTIME_COMPILE )
endif()

target_include_directories( rocfft-test
  PRIVATE
  ${rocfft-test_include_dirs}
  )

if( NOT BUILD_SHARED_LIBS )
  list(APPEND rocfft-test_link_libs ${ROCFFT_CLIENTS_HOST_LINK_LIBS} ${ROCFFT_CLIENTS_DEVICE_LINK_LIBS})
endif()

target_link_libraries( rocfft-test
  PRIVATE
  hip::device
  roc::rocfft
  hip::hiprand
  ${rocfft-test_link_libs}
  )

if( USE_CUDA )
  target_include_directories( rocfft-test
    PRIVATE
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocfft-test PRIVATE __HIP_PLATFORM_NVCC__ )
endif( )
target_link_libraries( rocfft-test PRIVATE ${ROCFFT_CLIENTS_HOST_LINK_LIBS} ${ROCFFT_CLIENTS_DEVICE_LINK_LIBS} )

option( BUILD_CLIENTS_TESTS_OPENMP "Build tests with OpenMP" ON )

if( BUILD_CLIENTS_TESTS_OPENMP )
  if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )  
    target_compile_options( rocfft-test PRIVATE -fopenmp )
    target_link_libraries( rocfft-test PRIVATE -fopenmp -L${HIP_CLANG_ROOT}/lib -Wl,-rpath=${HIP_CLANG_ROOT}/lib )
    target_include_directories( rocfft-test PRIVATE ${HIP_CLANG_ROOT}/include )
  else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options( rocfft-test PRIVATE -fopenmp=libomp )
      target_link_options( rocfft-test PRIVATE -fopenmp=libomp )
    endif()
  endif()
endif()

if(FFTW_MULTITHREAD)
  target_compile_options( rocfft-test PRIVATE -DFFTW_MULTITHREAD )
endif( )

set_target_properties( rocfft-test PROPERTIES
  DEBUG_POSTFIX "-d"
  CXX_STANDARD_REQUIRED ON
)

if( ROCFFT_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../staging" )
elseif( ROCFFT_CLIENTS_BUILD_SCOPE )
  set( TESTS_OUT_DIR "/../bin" )
else()
  set( TESTS_OUT_DIR "/bin" )
endif()
string( CONCAT TESTS_OUT_DIR "${PROJECT_BINARY_DIR}" ${TESTS_OUT_DIR} )

set_target_properties(rocfft-test
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY 
                      ${TESTS_OUT_DIR})
set_target_properties(rtc_helper_crash
                      PROPERTIES 
                      RUNTIME_OUTPUT_DIRECTORY 
                      ${TESTS_OUT_DIR})


rocm_install(TARGETS rocfft-test rtc_helper_crash COMPONENT tests)

if (WIN32)

  # Ensure tests run with HIP DLLs and not anything the driver owns
  # in system32.  Libraries like amdhip64.dll are also in the HIP
  # runtime, and we need run with those.  But the only way to make a
  # same-named DLL override something in system32 is to have it next
  # to the executable.  So copy them in.
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocfft-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} $<TARGET_FILE_DIR:rocfft-test> )
  endforeach( file_i )
endif()
