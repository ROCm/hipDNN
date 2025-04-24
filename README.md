# hipDNN

## Building
1. Grab or build a version of our development [Dockerfile](Dockerfile)
2. Run the dockerfile and mount the location of this repository.
3. Run cmake
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

# python frontend api
mkdir build
cd build
cmake -DHIP_DNN_FRONTEND_BUILD_PYTHON_BINDINGS=ON ..
```
4. Run your make commands(dont forget your -jnproc flag for using additional cores)
  - `make`: default build
  - `make check`: Builds everything and automatically runs the tests.
  - `make check_format`: checks the format of all c/c++ files in the repository and issues a warning
  - `make format`: formats any files in the repository which are not correctly formatted.
  - `make code_coverage`: builds, runs tests, and generates code coverage reports
    - Must add `-DCODE_COVERAGE=ON` to initial cmake command.

### Building the development docker container
Run `docker build -t <imagename> .` from the repository root


## Repo setup todo
- front end integration tests(todo, chat about this monday with team.)
    - Do we want integration tests how I set them up, do they make sense for the front end split out.
- create install package scaffolding

## CI Setup todo
- Ask to get coverage build setup
  - Make a warning occur if coverage for new code is < 80%.  Do not fail the build, to meet deadlines, we may have to take some shortcuts
- Ask to have release and debug builds occur
    - Also have tests run with these
- Ask to get clang format scanning as well.

## VSCode Setup

### CMake
- Install the cmake extension: 
  - Note: cmake should automatically find the compilers under /opt/rocm using the [ClangToolChain.cmake](./cmake/ClangToolChain.cmake) file. If you hit issues you can manually configure it below
    - Tell it to scan recursively for compilers and give it the /opt/rocm/llvm directory.
    - Once it has been found you can select CLANG instead of GCC.
  - If everything is setup correctly you can type Ctrl-Shift-B which will bring up a list of cmake configurations you can run.  You can also use F7 to build everything now. 

### Test Explorer
You can use https://github.com/matepek/vscode-catch2-test-adapter to have vscode test explorer auto load the gtest binaries into the editor

### Setting up Source Control Tooling 
To setup VSCode source control tooling to work correctly inside your development container you need to do the following.
1. On your local windows machine setup your ssh-agent: [setting-up-the-ssh-agent](https://code.visualstudio.com/docs/remote/troubleshooting#_setting-up-the-ssh-agent)
```
# Make sure you're running as an Administrator
Set-Service ssh-agent -StartupType Automatic
Start-Service ssh-agent
Get-Service ssh-agent

# add the ssh keys you need active
ssh-add C:\path\to\key1
ssh-add C:\path\to\key2
```
2. Add the following to your vscode settings.json file
```
"remote.SSH.useLocalServer": false, // having this set as true seems to cause random vscode hangs
"remote.SSH.enableAgentForwarding": true
```
3. you should now be able to fetch/pull from the repo using the built in source control tools without needing to enter a password all the time

#### Optional SSH Config
If you have multiple ssh keys you may need to setup a config file to help git know which key to use for a particular repo.  The file will look similar to below.
```
# Account 1
Host github_account_1
  HostName github.com
  User git
  IdentityFile ~/.ssh/primary_key_id_ed25519
  ForwardAgent yes

# Account 2
Host github_account_2
  HostName github.com
  User git
  IdentityFile ~/.ssh/secondary_key_id_ed25519
  ForwardAgent yes
```

In order to use this setup you will also need to clone your repository using the correct host name. 

```
# Default clone command
git clone git@github.com:AMD-ROCm-Internal/hipDNN.git

# Clone command using Host github_account_2.
git clone github_account_2:AMD-ROCm-Internal/hipDNN.git
```

Now when any git commands are ran in the repo cloned with github_account_2, git will automatically default to that key rather than the first one in the config file.