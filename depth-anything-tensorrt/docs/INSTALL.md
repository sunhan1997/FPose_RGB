## Installation Guide for C++

1. Install TensorRT using TensorRT official guidance.

    <details>
    <summary>Click here for Windows guide</summary>     
   
    1. Download the [TensorRT](https://developer.nvidia.com/tensorrt) zip file that matches the Windows version you are using.
    2. Choose where you want to install TensorRT. The zip file will install everything into a subdirectory called `TensorRT-8.x.x.x`. This new subdirectory will be referred to as `<installpath>` in the steps below.
    3. Unzip the `TensorRT-8.x.x.x.Windows10.x86_64.cuda-x.x.zip` file to the location that you chose. Where:
    - `8.x.x.x` is your TensorRT version
    - `cuda-x.x` is CUDA version `11.6`, `11.8` or `12.0`
    4. Add the TensorRT library files to your system `PATH`. To do so, copy the DLL files from `<installpath>/lib` to your CUDA installation directory, for example, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin`, where `vX.Y` is your CUDA version. The CUDA installer should have already added the CUDA path to your system PATH.
   
    </details>

    [Click here for installing tensorrt on Linux](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/install.md). 

2. Download and install any recent [OpenCV](https://opencv.org/releases/) for Windows. 
3. Modify TensorRT and OpenCV paths in CMakelists.txt:
   ```
   # Find and include OpenCV
   set(OpenCV_DIR "your path to OpenCV")
   find_package(OpenCV REQUIRED)
   include_directories(${OpenCV_INCLUDE_DIRS})
   
   # Set TensorRT path if not set in environment variables
   set(TENSORRT_DIR "your path to TensorRT")
   ```
  
4. Build project by using the following commands or  **cmake-gui**(Windows).

    1. Windows:
    ```bash
     mkdir build
    cd build
    cmake ..
    cmake --build . --config Release
    ```

    2. Linux(not tested):
    ```bash
    mkdir build
    cd build && mkdir out_dir
    cmake ..
    make
    ```

5. Finally, copy the opencv dll files such as `opencv_world490.dll` and `opencv_videoio_ffmpeg490_64.dll` into the `<depth_anything_installpath>/build/Release` folder.
    

## Tested Environment
   - TensorRT 8.6
   - CUDA 11.6
   - Windows 10
