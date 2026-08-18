# Build:
#   docker build -t llama-cpp-fp8 .
# Run (needs the host to have NVIDIA drivers + nvidia-container-toolkit):
#   docker run --rm -it --gpus all \
#       -v "$PWD":/workspace -w /workspace \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       llama-cpp-fp8

ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CUDA_DOCKER_ARCH=default
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl ca-certificates pkg-config \
      libssl-dev libgomp1 libcurl4-openssl-dev \
      python3 python3-pip python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Copy only what the C++ build needs. Editing scripts / assets / docs later
# will NOT invalidate this layer, so cmake stays cached across those edits.
COPY CMakeLists.txt CMakePresets.json ./
COPY cmake     ./cmake
COPY src       ./src
COPY ggml      ./ggml
COPY include   ./include
COPY common    ./common
COPY tools     ./tools
COPY examples  ./examples
COPY pocs      ./pocs
COPY vendor    ./vendor
COPY licenses  ./licenses
COPY grammars  ./grammars
COPY scripts   ./scripts

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
      export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        ${CMAKE_ARGS} && \
    cmake --build build --config Release -j"$(nproc)"

ENV PATH="/app/build/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/build/bin:${LD_LIBRARY_PATH}"

# Copy only the requirements files. Editing scripts / assets later won't
# invalidate the pip install layer either.
COPY requirements.txt ./
COPY requirements     ./requirements

# Install CUDA-capable torch first so that requirements.txt (which pulls the
# CPU wheel via --extra-index-url) will see torch already satisfied and skip it.
RUN pip3 install --break-system-packages --no-cache-dir --upgrade \
        pip setuptools wheel && \
    pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch~=2.6.0" && \
    pip3 install --break-system-packages --no-cache-dir \
        -r requirements.txt && \
    pip3 install --break-system-packages --no-cache-dir \
        "llmcompressor<0.12" "transformers<5"

# No file dependency: safe to keep cached across script/asset changes.
# scipy is not directly used, but transformers 4.57's generation.candidate_generator
# imports sklearn at module load; sklearn then imports scipy. Missing scipy therefore
# breaks AutoTokenizer / AutoModel imports entirely.
# Pin scipy<1.14 because llama.cpp's requirements pin numpy~=1.26.4, and scipy>=1.14
# requires numpy>=2.0 — otherwise sklearn refuses to load with a version-mismatch error.
RUN pip3 install --break-system-packages --no-cache-dir \
        "huggingface_hub[cli,hf_transfer]" \
        "scipy<1.14"

# Mamba / Nemotron-H hybrid models require mamba_ssm + causal-conv1d for their
# custom kernels. These packages compile CUDA code against the installed torch,
# so --no-build-isolation is required (else pip's ephemeral build env has no
# torch/CUDA). Its own layer: compile is slow (~10 min) but rarely changes.
RUN pip3 install --break-system-packages --no-cache-dir --no-build-isolation \
        "causal-conv1d>=1.4.0" \
        mamba-ssm

# vLLM for FP8 serving and for the test_conversion_fp8 --vllm cross-check.
# ~2 GB of wheels + deps, no file dependency, so editing scripts / assets
# never re-triggers this download.
# Pin vllm<0.24: from 0.24 onward vllm requires transformers>=5.5.3, which
# would upgrade transformers to v5 and break llmcompressor (pinned to
# transformers<=4.57.6). Re-assert transformers<5 on the same install so
# pip can't silently override the earlier constraint.
RUN pip3 install --break-system-packages --no-cache-dir \
        "vllm<0.24" "transformers<5"

# Ollama binary (needed by test_conversion/test_main.py to compare the GGUF
# against the original HF model). test.py auto-starts `ollama serve` in the
# background when it isn't already reachable, provided this binary is on PATH.
# Placed as the last install layer. Ollama publishes as .tar.zst (zstd) since
# ~v0.31 — no more .tgz — hence the apt install of zstd and tar's --zstd flag.
# Rewrite `http://` to `https://` on Ubuntu mirrors before any apt call, so
# `apt-get update` works from networks that block plain-HTTP egress (common in
# managed environments). Also bundles `less` (interactive log paging) into the
# same apt hop as `zstd` so we don't need a second network round-trip.
RUN find /etc/apt -type f \( -name '*.sources' -o -name '*.list' \) -exec sed -i \
        -e 's|http://ports.ubuntu.com|https://ports.ubuntu.com|g' \
        -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
        {} + \
 && apt-get update && apt-get install -y --no-install-recommends zstd less \
 && rm -rf /var/lib/apt/lists/* \
 && arch=$(uname -m) \
 && case "$arch" in x86_64) o=amd64 ;; aarch64) o=arm64 ;; *) echo "unsupported arch: $arch" >&2; exit 1 ;; esac \
 && curl -fsSL "https://ollama.com/download/ollama-linux-${o}.tar.zst" | tar --zstd -xf - -C /usr

# Everything else (Python scripts, assets, README, etc.). Placed last so that
# editing any of these only rebuilds this small layer, not cmake or pip.
COPY . .

WORKDIR /workspace

CMD ["/bin/bash"]
