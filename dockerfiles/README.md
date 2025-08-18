# Dockerfiles

## Building the Dockerfiles

You can build either dockerfile using the following command:

```bash
docker build -f <Dockerfile path> -t <container_tag> .
```

## Available Containers

### Ubuntu 22.04
- Development container for Ubuntu 22.04
- [Dockerfile.ubuntu22](Dockerfile.ubuntu22) - Ubuntu 22.04 environment

### AlmaLinux 8
- [Dockerfile.almalinux](Dockerfile.almalinux) - RHEL8 environment
- Container to support builds using ROCm on RHEL-based systems
