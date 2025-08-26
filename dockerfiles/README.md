# hipDNN Docker Environments

This directory contains Dockerfiles for building hipDNN development environments on different platforms.

## 📋 Prerequisites

- Docker installed on your system
- ROCm-compatible GPU (for running with GPU support)
- Sufficient disk space for Docker images

## 🐳 Available Containers

### Ubuntu 22.04

- **File**: [`Dockerfile.ubuntu22`](Dockerfile.ubuntu22)
- **Base Image**: Ubuntu 22.04 LTS
- **Purpose**: Primary development environment for hipDNN
- **Includes**:
  - ROCm development tools
  - CMake build system
  - Google Test framework
  - Ninja build tool

### AlmaLinux 8

- **File**: [`Dockerfile.almalinux`](Dockerfile.almalinux)
- **Base Image**: AlmaLinux 8 (RHEL8 compatible)
- **Purpose**: Support for RHEL-based systems and enterprise environments
- **Includes**:
  - ROCm development tools
  - CMake build system
  - Google Test framework
  - Ninja build tool

## 🔨 Building the Docker Images

### Basic Build Command

```bash
docker build -f <Dockerfile> -t <image-name> .
```

### Examples

Build Ubuntu 22.04 container:
```bash
docker build -f Dockerfile.ubuntu22 -t hipdnn:ubuntu22 .
```

Build AlmaLinux 8 container:
```bash
docker build -f Dockerfile.almalinux -t hipdnn:almalinux8 .
```

## 🚀 Running the Containers

### Basic Run Command

To run the hipDNN docker environment with full GPU support:

```bash
docker run -it \
  -v /path/to/hipdnn/repo:/workspace/hipdnn \
  --privileged \
  --rm \
  --device=/dev/kfd \
  --device /dev/dri:/dev/dri:rw \
  --volume /dev/dri:/dev/dri:rw \
  -v /var/lib/docker/:/var/lib/docker \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  <image-name>
```

### Command Options Explained

| Option | Purpose |
|--------|---------|
| `-it` | Interactive terminal |
| `-v /path/to/hipdnn/repo:/workspace/hipdnn` | Mount hipDNN source code |
| `--privileged` | Required for GPU access |
| `--rm` | Remove container after exit |
| `--device=/dev/kfd` | AMD GPU compute device |
| `--device /dev/dri:/dev/dri:rw` | Direct Rendering Infrastructure for GPU |
| `--group-add video` | Add container to video group for GPU access |
| `--cap-add=SYS_PTRACE` | Enable debugging capabilities |
| `--security-opt seccomp=unconfined` | Required for some debugging tools |

### Example Run Commands

Ubuntu 22.04:
```bash
docker run -it \
  -v $(pwd):/workspace/hipdnn \
  --privileged \
  --rm \
  --device=/dev/kfd \
  --device /dev/dri:/dev/dri:rw \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  hipdnn:ubuntu22
```

AlmaLinux 8:
```bash
docker run -it \
  -v $(pwd):/workspace/hipdnn \
  --privileged \
  --rm \
  --device=/dev/kfd \
  --device /dev/dri:/dev/dri:rw \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  hipdnn:almalinux8
```

## 💡 Tips and Best Practices

1. **Volume Mounting**: Mount your hipDNN source code directory to persist changes:
   ```bash
   -v /absolute/path/to/hipdnn:/workspace/hipdnn
   ```

2. **GPU Verification**: Once inside the container, verify GPU access:
   ```bash
   rocm-smi
   ```

3. **Development Workflow**: The containers include all necessary development tools, so you can edit code on your host machine and build/test inside the container.

## 📁 Helper Scripts

The `scripts/` directory contains helper scripts used during Docker image building:

- `install_cmake.sh` - Installs CMake build system
- `install_gtest.sh` - Installs Google Test framework
- `install_ninja.sh` - Installs Ninja build tool

## ⚠️ Troubleshooting

### GPU Not Detected
- Ensure ROCm is properly installed on the host system
- Verify that `/dev/kfd` and `/dev/dri` exist on the host
- Check that your user is in the `video` and `render` groups

### Permission Issues
- Make sure to use the `--privileged` flag
- Verify that the mounted directories have appropriate permissions

### Build Failures
- Ensure all submodules are initialized: `git submodule update --init --recursive`
- Check that the ROCm version in the container matches your GPU requirements
