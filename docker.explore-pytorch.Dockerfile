FROM pytorch/pytorch:latest
# FROM --platform=linux/amd64 ubuntu:25.10

# ENV DEBIAN_FRONTEND=noninteractive

# Set UTF-8 locale
# RUN apt-get update && apt-get install -y locales && \
#     locale-gen en_US.UTF-8
# ENV LANG=en_US.UTF-8
# ENV LANGUAGE=en_US:en
# ENV LC_ALL=en_US.UTF-8

# Install system packages
RUN apt-get update && apt-get install -y \
    zsh git curl wget \
    && apt-get clean

# Set Zsh as default shell
SHELL ["/usr/bin/zsh", "-c"]

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

CMD ["zsh"]

# Set working directory to match host project folder
WORKDIR /root/project
