# Building hipDNN 
This guide assumes you are using the development dockerfile to build and run hipDNN.  If you are not, you will need
to ensure your environment is setup correctly.

## System Dependencies
The development dockerfile is a good place to look to understand what system dependencies are required in order build
hipDNN.  At the time of writing this readme, the following bare minimum system dependencies are required:
- Package Repository: ROCm 6.4 or later
    - rocm-llvm-devel (amd clang compiler is included with this.)
    - hip-devel
- Python3
- Cmake 3.25.2 or later
- Ninja 1.12.1 or later
    - Ninja is the default build system for hipDNN.
    - If you dont set the environment variable `CMAKE_GENERATOR` to `Ninja`, cmake will default to using make.
- 


## 3rdParty Dependencies
The current open source dependencies used in this repository are
- flatbuffers
- gtest
- spdlog

These can be found in our [Dependencies.cmake file](../cmake/Dependencies.cmake).  By default, cmake will look
for these dependencies installed on the system before it attempts to pull them from a remote source.

## Quickstart Building and Installing HipDNN
1. Build a version of our development [Dockerfile](../Dockerfile). See [building docker](#build-the-development-docker-container)
2. Run the docker image you built and mount the location of this repository.
    - `docker run -it -v $HOME:/location/of/repository --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined docker_image_name`
3. Use Cmake to generate the make files 
```
# default release build
cd path/to/repository
mkdir build
cd build
cmake ..
```
4. Build and install using Ninja:  `ninja -j10 install`
    - By default the install location will be in the `/opt/rocm` folder.

### Additional Build Configurations and Targets
#### Cmake Configurations
```
# default release
mkdir build
cd build
cmake ..

# default debug
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# code coverage
mkdir build
cd build
cmake -DCODE_COVERAGE=ON ..

# python frontend api (requires pybind11 preinstallation)
mkdir build
cd build
cmake -DHIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON ..
```

#### Ninja Targets
When building with ninja dont forget to add the -jnproc flag for using additional cores
  - `ninja`: default build
  - `ninja install`: installs the library to the default location (/opt/rocm)
  - `ninja check`: Builds everything and automatically runs the tests via the ctest runner.
  - `ninja check_format`: checks the format of all c/c++ files in the repository and issues a warning
  - `ninja format`: formats any files in the repository which are not correctly formatted.
  - `ninja code_coverage`: builds, runs tests, and generates code coverage reports
    - Must add `-DCODE_COVERAGE=ON` to initial cmake command.


### Build the development docker container
The development dockerfile is here: [Dockerfile](../Dockerfile).

To build the docker file, all you need to do is navigate to the repository root 
Run `docker build -t <imagename> .` from the repository root

## Building Samples
todo