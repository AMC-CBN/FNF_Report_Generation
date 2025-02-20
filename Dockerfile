FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# 시간대 설정 자동화
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# NVIDIA GPG 키 추가
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-dri \
    libglx0 \
    libopengl0 \
    python3-opencv \
    mesa-utils \
    x11-xserver-utils \
    && ldconfig

# NVIDIA GL 라이브러리 링크
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},display

# 기본 패키지 업그레이드
RUN pip install --upgrade pip
RUN pip install --upgrade typing-extensions jinja2

# 패키지 설치
RUN pip install numpy==1.20.3 \
    pandas==1.3.0 \
    opencv-python-headless==4.5.1.48 \
    pydicom>=2.0.0 \
    scikit-learn>=0.24.0 \
    tqdm>=4.50.0 \
    scipy>=1.5.0 \
    torch>=1.7.0 \
    openpyxl>=3.0.0 \
    seaborn \
    matplotlib

# Jupyter 관련 패키지 설치
RUN pip install --upgrade \
    typing-extensions>=4.1.0 \
    jinja2>=3.0.3 \
    jupyter-core>=4.12.0 \
    jupyter-client==7.4.9 \
    notebook==6.4.12 \
    jupyter-server>=2.4.0,\<3.0.0 \
    ipython==7.34.0 \
    jupyterlab==4.3.5

WORKDIR /workspace/FNF

