# Clean-machine proof for Linux: build llama-server from a pinned tag, install
# local-llm with install.sh (uv path), and run the whole flow as a normal user.
FROM ubuntu:24.04 AS llama
ARG LLAMA_TAG=b10256
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp /llama.cpp
RUN cmake -S /llama.cpp -B /llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=ON \
    && cmake --build /llama.cpp/build --target llama-server -j "$(nproc)"

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git jq procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
RUN useradd -m -s /bin/bash tester
USER tester
WORKDIR /home/tester
ENV PATH="/home/tester/.local/bin:${PATH}"
COPY --chown=tester:tester . /home/tester/src
ENV LOCAL_LLM_SOURCE=/home/tester/src
RUN sh /home/tester/src/install.sh --uv -y
ENTRYPOINT ["/bin/bash", "/home/tester/src/tests/e2e/inside.sh"]
