cmake_minimum_required(VERSION 3.10)

project(CustomKrylovSolvers VERSION 0.1)
include(CTest)

find_package(universal CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(EigenUniversalIntegration 0.1.0 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEBUG   "Enable Debugging"               OFF)
option(OPENMP  "Enable OpenMP"                  ON)

if(DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else(DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native") # add -DNDEBUG later when everything works

    if(OPENMP)
        message("using openmp")
        find_package(OpenMP REQUIRED)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    endif(OPENMP)

endif(DEBUG)

message("Flags: ${CMAKE_CXX_FLAGS}")

list(APPEND USE_Type_Flags USE_Float USE_Double USE_LongDouble USE_Posit16 USE_Posit32 USE_Posit64)
list(APPEND USE_Algorithm_Flags USE_GMRES USE_QMR USE_QMRWLA)

list(APPEND GMRESRunners FloatGMRES DoubleGMRES LongDoubleGMRES Posit16GMRES Posit32GMRES Posit64GMRES)
list(APPEND QMRRunners FloatQMR DoubleQMR LongDoubleQMR Posit16QMR Posit32QMR Posit64QMR)
list(APPEND QMRWLARunners FloatQMRWLA DoubleQMRWLA LongDoubleQMRWLA Posit16QMRWLA Posit32QMRWLA Posit64QMRWLA)

list(APPEND RUNNERS GMRESRunners QMRRunners QMRWLARunners)

foreach(runners algorithm IN ZIP_LISTS RUNNERS USE_Algorithm_Flags)
    foreach(runner type IN ZIP_LISTS ${runners} USE_Type_Flags)
        add_executable(${runner} src/main.cxx)
        target_link_libraries(${runner} Eigen3::Eigen OpenMP::OpenMP_CXX EigenUniversalIntegration::EigenUniversalIntegration)
        target_include_directories(${runner} PUBLIC "${PROJECT_BINARY_DIR}" "include" universal)
        target_compile_definitions(${runner} PUBLIC ${algorithm} ${type})
        install(TARGETS ${runner} DESTINATION bin)
    endforeach()
endforeach()
