# hipDNN Docker Environments

This directory contains the Dockerfile for building hipDNN development environment.

## 📋 Prerequisites

- Docker installed on your system
- ROCm-compatible GPU (for running with GPU support)
- Sufficient disk space for Docker images

## 🐳 Ubuntu 24.04

- **File**: [`Dockerfile.ubuntu24`](Dockerfile.ubuntu24)
- **Base Image**: Ubuntu 24.04 LTS
- **Purpose**: Primary development environment for hipDNN
- **Includes**:
  - ROCm development tools
  - CMake build system
  - Google Test framework
  - Ninja build tool

## 🔨 Building the Docker Image

The Dockerfile supports two build types: **prebuilt** (using nightly tarballs) and **fullbuild** (building from source).

### Build Arguments

#### 🔧 Common Arguments (Both Build Types)

| Argument | Default | Description | Valid Values |
|----------|---------|-------------|--------------|
| `BUILD_TYPE` | `prebuilt` | Selects build method | `prebuilt`, `fullbuild` |
| `THEROCK_ASIC` | `gfx94X` (prebuilt)<br>`gfx90a` (fullbuild) | GPU architecture target | `gfx94X`, `gfx950`, `gfx110X`, `gfx90a`, etc. |

#### 📦 Prebuilt-Only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_GIT_TAG` | `7.0.0rc20250909` | Git tag for [nightly tarballs](https://therock-nightly-tarball.s3.amazonaws.com/) |

> **Note**: Prebuilt mode downloads pre-compiled binaries from TheRock nightly builds (much faster)

#### 🏗️ Fullbuild-Only Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_GIT_HASH` | `default` | Specific git commit hash to checkout (uses default branch if not specified) |

> **Note**: Fullbuild mode clones and compiles TheRock from source (slower but more flexible)

### Build Examples

#### 📦 Prebuilt Mode (Recommended)

**Default prebuilt** (gfx94X, tag 7.0.0rc20250909):
```bash
docker build -f Dockerfile.ubuntu24 -t hipdnn:prebuilt .
```

**Override ASIC and Git Tag** (gfx950, tag 7.0.0rc20250908):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg THEROCK_ASIC=gfx950 --build-arg THEROCK_GIT_TAG=7.0.0rc20250908 -t hipdnn:prebuilt_gfx950 .
```

#### 🏗️ Fullbuild Mode

**Default fullbuild** (gfx90a, default branch):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild -t hipdnn:fullbuild .
```

**Custom ASIC and Git Hash** (gfx950, hash abcd1234):
```bash
docker build -f Dockerfile.ubuntu24 --build-arg BUILD_TYPE=fullbuild --build-arg THEROCK_ASIC=gfx950 --build-arg THEROCK_GIT_HASH=abcd1234 -t hipdnn:custom_source_build .
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
  hipdnn:prebuilt
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
