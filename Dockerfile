FROM lnfu/robot-prod:ubuntu22.04-ros2-humble AS builder

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-dev-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/HCIS-Lab/home-interfaces.git /workspace/src/home_interfaces

COPY . /workspace/src/home_nlp

RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build

FROM lnfu/robot-prod:ubuntu22.04-ros2-humble

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY --from=builder /workspace/install /workspace/install
