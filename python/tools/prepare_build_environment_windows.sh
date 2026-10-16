#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
curl --netrc-optional -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/13.0.3/local_installers/cuda_13.0.3_windows.exe
# CUDA 13 moved the compiler headers (crt) and NVVM out of the nvcc subpackage.
./cuda.exe -s nvcc_13.0 crt_13.0 nvvm_13.0 thrust_13.0 cudart_13.0 cublas_dev_13.0 curand_dev_13.0

rm cuda.exe

# See https://github.com/oneapi-src/oneapi-ci for installer URLs
if [ ! -d "C:/Program Files (x86)/Intel/oneAPI" ]; then
    curl --netrc-optional -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1f18901e-877d-469d-a41a-a10f11b39336/intel-oneapi-base-toolkit-2025.3.0.372_offline.exe
    ./webimage.exe -s -x -f webimage_extracted --log extract.log
    rm -f webimage.exe || true
    ./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.
fi

NPROC=$(nproc)
ONEDNN_VERSION=3.1.1
if [ ! -d "C:/Program Files (x86)/oneDNN" ]; then
    curl --netrc-optional -L -O https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
    tar xf *.tar.gz && rm *.tar.gz
    cd oneDNN-*
    cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
    cmake --build . --config Release --target install --parallel $NPROC
    cd ..
    rm -r oneDNN-*
fi

mkdir build
cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CTRANSLATE2_ROOT -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win;C:/Program Files (x86)/oneDNN" -DBUILD_CLI=OFF -DWITH_DNNL=ON -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" -DCUDA_DYNAMIC_LOADING=ON -DCUDA_NVCC_FLAGS="-Xfatbin=-compress-all" -DCUDA_ARCH_LIST="Common" ..
cmake --build . --config Release --target install --parallel $NPROC --verbose
cd ..
rm -r build

cp README.md python/
cp $CTRANSLATE2_ROOT/bin/ctranslate2.dll python/ctranslate2/
cp "C:/Program Files (x86)/Intel/oneAPI/2025.3/bin/libiomp5md.dll" python/ctranslate2/
