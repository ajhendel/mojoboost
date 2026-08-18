# hardware/Dockerfile - a hermetic box for one validation run, any vendor.
#
# STATUS: NEVER BUILT. This file has not been built, has not been run, and no
# image from it has ever seen a GPU. It follows this project's own rule that
# support and validation are different claims, so it is committed as a
# `designed` artifact and nothing here should be quoted as working. The first
# person to build it should say so in a hardware result, and this header should
# then say what happened instead of saying this. The base image tags below are
# plausible rather than verified; if one does not resolve, pick the nearest tag
# your provider offers and record which one you used.
#
# WHY THIS EXISTS. `hardware/run_validation.sh` already needs no root and
# installs nothing but pixi's own environment, so on a machine whose driver
# userspace is already correct you do not need this file at all and should not
# use it. What this buys is the case where the userspace is *not* already
# correct, most often a rented box with a bare OS image and no ROCm or CUDA
# libraries, where installing them on the host is the only step in the whole
# procedure that wants root.
#
# BUILD
#   docker build -f hardware/Dockerfile -t mojotrees-validate .
#   docker build -f hardware/Dockerfile -t mojotrees-validate \
#       --build-arg BASE=rocm/dev-ubuntu-24.04:6.3.3-complete .
#
# RUN, from a checkout, which is mounted rather than baked in so the image
# never pins a stale commit.
#   NVIDIA:
#     docker run --rm --gpus all -v "$PWD:/work" mojotrees-validate
#   AMD:
#     docker run --rm --device=/dev/kfd --device=/dev/dri \
#         --group-add video --group-add render \
#         --security-opt seccomp=unconfined \
#         -v "$PWD:/work" mojotrees-validate
#
# The driver itself lives on the host and is not installed here. Mojo's floors
# are NVIDIA 580 or later and AMD 6.3.3 or later; a container cannot raise them.
#
# IF THE RUN FAILS INSIDE THIS CONTAINER, that is not automatically a mojotrees
# result. GPU passthrough is its own failure mode and this file has never been
# tested. Re-run `sh hardware/run_validation.sh` on the bare host before
# reporting anything, and if the two disagree, the disagreement is the finding.

ARG BASE=nvidia/cuda:12.8.0-devel-ubuntu24.04
FROM ${BASE}

ARG BASE
LABEL org.opencontainers.image.source="https://github.com/mojotrees/mojotrees" \
      org.opencontainers.image.description="mojotrees GPU validation runner (untested)" \
      com.mojotrees.base="${BASE}" \
      com.mojotrees.status="never-built"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# pixi resolves the Mojo toolchain from pixi.lock, so the toolchain version is
# pinned by the checkout you mount and not by this image.
ENV PIXI_HOME=/opt/pixi
ENV PATH=/opt/pixi/bin:$PATH
RUN curl -fsSL https://pixi.sh/install.sh | sh

# The validation scripts promise they need no root. Honor that here so a run in
# this container exercises the same path a contributor runs on a bare host.
RUN useradd --create-home --uid 1000 validator \
    && install -d -o validator -g validator /work
USER validator
WORKDIR /work

CMD ["sh", "hardware/run_validation.sh"]
