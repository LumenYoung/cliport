FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID


RUN apt-key del A4B469963BF863C \
  && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
  && apt-get update 

  # && dpkg -i cuda-keyring_1.0-1_all.deb \
  # apt-get install wget \
  # && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \

RUN apt-get install -y sudo wget
RUN useradd -ms /bin/bash $USER_NAME
RUN usermod -aG sudo $USER_NAME
RUN yes $USER_PASSWORD | passwd $USER_NAME

# set uid and gid to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# work directory
WORKDIR /home/$USER_NAME

# install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

COPY ./scripts/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

# setup python environment
RUN cd $WORKDIR

# install python requirements
# RUN sudo python3 -m pip install --upgrade pip && \ 
#     sudo python3 -m pip install --upgrade

# install pip3
RUN apt-get -y install python3-pip
RUN sudo python3 -m pip install --upgrade pip

# install pytorch
RUN sudo pip3 install --no-input\
   torch==1.9.1+cu111 \
   torchvision==0.10.1+cu111 \
   -f https://download.pytorch.org/whl/torch_stable.html

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils \
   python3-setuptools \
   python3-dev \
   && rm -rf /var/lib/apt/lists/*

RUN sudo pip3 install --no-input \
   absl-py>=0.7.0  \
   gym==0.17.3 \
   pybullet>=3.0.4 \
   matplotlib>=3.1.1 \
   opencv-python>=4.1.2.30 \
   meshcat>=0.0.18 \
   scipy==1.4.1 \
   scikit-image==0.17.2 \
   pytorch_lightning==1.0.3 \
   tdqm \
   hydra-core==1.0.5 \
   wandb \
   transformers==4.3.2 \
   kornia \
   ftfy \
   regex \
   ffmpeg \
   imageio-ffmpeg \
   packaging==21.3 \
   chafa.py \
   langchain \
   transforms3d 

RUN pip3 uninstall --no-input -y wandb \
  && pip3 install --no-input wandb

# change ownership of everything to our user
RUN mkdir -p /home/$USER_NAME/cliport
RUN cd /home/$USER_NAME/cliport && echo $(pwd) && chown $USER_NAME:$USER_NAME -R .
RUN echo "export CLIPORT_ROOT=~/cliport" >> /home/$USER_NAME/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:~/cliport" >> /home/$USER_NAME/.bashrc
RUN echo "cd cliport && python3 setup.py develop" >> /home/$USER_NAME/.bashrc
