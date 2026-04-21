FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-tk \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libglfw3 \
    libglew2.1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /opt/venv

RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel
RUN /opt/venv/bin/pip install \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pyyaml==6.0.2 \
    pillow==10.4.0 \
    imageio==2.35.1 \
    ninja==1.11.1.1

RUN /opt/venv/bin/pip install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Copy local Isaac Gym Preview 4 archive and extract inside Linux to preserve symlinks.
COPY tools/vendor/IsaacGym_Preview_4_Package.tar.gz /tmp/isaacgym.tar.gz
RUN mkdir -p /opt && tar -xzf /tmp/isaacgym.tar.gz -C /opt && rm -f /tmp/isaacgym.tar.gz

# Install Isaac Gym core package from local archive extraction.
RUN /opt/venv/bin/pip install /opt/isaacgym/python
RUN mkdir -p /opt/venv/lib/python3.8/site-packages/isaacgym/_bindings/src && \
    cp -r /opt/isaacgym/python/isaacgym/_bindings/src/gymtorch \
    /opt/venv/lib/python3.8/site-packages/isaacgym/_bindings/src/

# Install runtime helpers used by IsaacGymEnvs while preserving Isaac Gym torch pin.
RUN /opt/venv/bin/pip install hydra-core==1.3.2
RUN /opt/venv/bin/pip install --no-deps rl-games==1.6.5
RUN /opt/venv/bin/pip install gym==0.23.1 cloudpickle==2.2.1
RUN /opt/venv/bin/pip install \
    omegaconf==2.3.0 \
    termcolor==2.4.0 \
    jinja2==3.1.6 \
    pyvirtualdisplay==3.0 \
    urdfpy==0.0.22 \
    pysdf==0.1.9 \
    warp-lang==0.10.1 \
    trimesh==3.23.5

# IsaacGymEnvs packaging omits some task directories from built wheels;
# editable install keeps the full source tree available in site import path.
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git /opt/IsaacGymEnvs && \
    cd /opt/IsaacGymEnvs && \
    git checkout aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3 && \
    /opt/venv/bin/pip install --no-deps -e /opt/IsaacGymEnvs

WORKDIR /workspace

CMD ["/bin/bash", "-lc", "sleep infinity"]
