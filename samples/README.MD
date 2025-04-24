# hipDNN-samples
## How to build
- In the main project directory (root for hipDNN above this folder) run the following commands to build & install hipDNN:
    - `mkdir build`
    - `cd build`
    - `cmake ..`
    - `make install -j`
- The above steps will install the hipdnn_frontend & hipdnn_sdk headers into /opt/rocm/include
- Once those are installed, then inside the samples directory run the following commands:
    - `mkdir build`
    - `cd build`
    - `cmake ..`
    - `make -j`
- inside the build folder will be hipdnn_samples which can now be ran

## Client samples
- Currently the client samples are a PoC that demonstrates using the header only installs from hipdnn_frontend & hipdnn_sdk to create a client application
- In the future we will be looking to add more useful samples that demonstrate integrated functionality, and also should include the ability to call find_package(hipdnn_frontend) & find_package(hipdnn_sdk) to set up the targets properly.