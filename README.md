<p align="center">
  <img alt="PhageGAP" width="256px" height="256px" src="https://github.com/Integrative-Transcriptomics/PhageGAP/blob/b4cdcb101a42fc9cd9f42937b88f2475d8d59784/app/phagegap/static/media/phagegap-logo-transparent-1024.png?raw=true">
</p>

PhageGAP is a [machine-learning framework](https://github.com/Integrative-Transcriptomics/PhageGAP-model) provided as an [interactive web application](https://phagegap.cs.uni-tuebingen.de/) for functional annotation of bacteriophage proteins. It combines protein-language-model embeddings with a supervised convolutional neural network to predict 89 curated functional classes, and to provide functional hypotheses for proteins that cannot be characterized reliably by sequence homology alone.

The web application complements classification with an interactive reference protein t-SNE landscape, nearest-neighbor analysis, sequence comparison, predicted protein structures, and genomic context.

## Architecture

PhageGAP consists of two independently deployed services:

- **Inference service** — GPU-backed prediction service running in an Apptainer container. It performs ProtT5 embedding, CNN classification, PCA transformation, t-SNE projection, and nearest-neighbor analysis. The respective source code is provided in the `inference/` subdirectory of this repository.
- **Web application** — Dockerized Flask application providing the browser interface, reference metadata and structures, sequence comparison, and communication with the inference service. The respective source code is provided in the `app/` subdirectory of this repository.

The trained model, dimensionality-reduction models, reference metadata, predicted structures, and pre-built application images are distributed separately through Zenodo (see [Releases](https://github.com/Integrative-Transcriptomics/PhageGAP/releases)).

**This repository contains the source code for both services, as well as instructions for building and deploying them.**

## Requirements

For the pre-built deployment:

- Docker (version ≥28.3.2).
- Apptainer (version ≥1.4.5).
- NVIDIA GPU with a compatible CUDA driver. The distributed image was built using a CUDA 12.8 software environment.
- Apptainer NVIDIA support (`--nv`).
- The PhageGAP model and reference files are from an associated release.

### Required Files

Download the following files from the PhageGAP release:

```text
cnn_model1_final.pt
pca.joblib
tsne.joblib

phagegap-metadata.tsv.gz
phagegap-structures-metadata.tsv.gz
phagegap-structures.tar.gz

phagegap-inference-1.0.0.sif
phagegap-app-1.0.0.tar.gz
```

A convenient runtime layout is for example:

```text
phagegap-runtime/
├── models/
│   ├── cnn_model1_final.pt
│   ├── pca.joblib
│   └── tsne.joblib
├── metadata/
│   ├── phagegap-metadata.tsv.gz
│   └── phagegap-structures-metadata.tsv.gz
├── structures/
│   └── ...
├── secrets/
│   ├── phagegap-api-token
│   └── phagegap-api-key
├── config.toml
├── phagegap-inference-1.0.0.sif
└── phagegap-app-1.0.0.tar.gz
```

Extract `phagegap-structures.tar.gz` into `structures/`.

## Configuration

Edit or create the `config.toml` file:

```toml
title = "PhageGAP classifier/inference service 1.0.0 configuration."

[app]
max_content_length = 500_000
max_form_memory_size = 500_000
max_form_parts = 1000
api_token_path = "secrets/phagegap-api-token"

[setup]
seed = 42
device = "cuda"

[embed]
model_type = "prot_t5"
pool_layers = []
pool_strategy = "mean"

[predict]
model_path = "models/cnn_model1_final.pt"
model_type = "cnn"

[manifold]
pca_path = "models/pca.joblib"
tsne_path = "models/tsne.joblib"
```

#### Generate API Token

PhageGAP implements a CRSF protocol for communication between the web application and inference service. The `api_token_path` in the configuration file specifies the location of a shared secret token that both services must use to authenticate requests. The API key is used for signing session cookies in the web application.

Create two independent random secrets of at least 32 characters, e.g., using OpenSSL:

```bash
mkdir -p secrets
openssl rand -hex 32 > secrets/phagegap-api-token
openssl rand -hex 32 > secrets/phagegap-api-key
chmod 600 secrets/phagegap-api-token secrets/phagegap-api-key
```

#### Select Model and Device

While the PhageGAP framework is agnostic to the model used for inference, the distributed release is restricted to a CNN classifier trained on ProtT5 embeddings.

We highly recommend running the inference service on a GPU. The `device` option in the configuration file can be set to `"cuda"` or `"cpu"`. If you do not have a GPU, you can set it to `"cpu"`, but inference will be significantly slower.

## Start the Inference Service

From `phagegap-runtime/`:

```bash
apptainer instance start \
    --nv \
    --env CUDA_VISIBLE_DEVICES="0" \
    --bind "$PWD/models:/app/models:ro" \
    --bind "$PWD/secrets:/app/secrets:ro" \
    --bind "$PWD/config.toml:/app/config.toml:ro" \
    phagegap-inference-1.0.0.sif \
    phagegap-inference
```

The inference service must be reachable from the web application container. The examples below assume that it listens on port `20101` on the Docker host.

## Start the Web Application

Load the distributed Docker image:

```bash
docker --input phagegap-app-1.0.0.tar.gz
```

Run the application:

```bash
docker run \
    --restart unless-stopped \
    --cpus="8" \
    --memory="64g" \
    --add-host=host.docker.internal:host-gateway \
    -e INFERENCE_SERVICE_URL="http://host.docker.internal:20101" \
    -v "$PWD/structures:/app/structures:ro" \
    -v "$PWD/metadata:/app/phagegap/static/data:ro" \
    -v "$PWD/secrets:/app/secrets:ro" \
    -p 127.0.0.1:20102:5001/tcp \
    phagegap-app
```

The web application is then available at:

```text
http://localhost:20102
```

For deployments on separate hosts, replace `INFERENCE_SERVICE_URL` with the reachable address of the inference service. We recommend restricting access to that service at the network or firewall level.

## Building from Source

To build the inference service image from the repository root:

```bash
apptainer build \
    inference/images/phagegap-inference-1.0.0.sif \
    inference/image.def
```

To build the web application:

```bash
docker build --tag phagegap-app app
```

The resulting services can then be started using the deployment procedure above.

## License

The PhageGAP source code is distributed under the GNU General Public License version 3.0 or later.

The PhageGAP model weights, fitted dimensionality-reduction models, reference metadata, and distributed structure data are licensed separately under CC BY-NC 4.0.

Third-party software and dependencies retain their respective licenses. See the license information accompanying the release for details.

## Citation

> Hackl, S., Scheurenbrand, M., Nieselt, K., & Wolfram-Schauerte, M. (2026). PhageGAP (1.0.0) [Unpublished software]. GitHub. [github.com/Integrative-Transcriptomics/PhageGAP](https://github.com/Integrative-Transcriptomics/PhageGAP)
