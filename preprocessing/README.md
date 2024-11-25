## Preprocessing
This folder contains the overall preprocessing for the considered subreddits.


## Setup
### Preliminaries

It's recommended to use [`mamba`](https://github.com/mamba-org/mamba) to manage dependencies. `mamba` is a drop-in replacement for `conda` re-written in C++ to speed things up significantly (you can stick with `conda` though). To provide reproducible environments, we use `conda-lock` to generate lockfiles for each platform.

<details><summary>Installing <code>mamba</code></summary>

<p>

On Unix-like platforms, run the snippet below. Otherwise, visit the [mambaforge repo](https://github.com/conda-forge/miniforge#mambaforge). Note this does not use the Anaconda installer, which reduces bloat.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

</details>

1. If you have not yet installed Docker, please install it under: (https://docs.docker.com/get-started/get-docker/)
1. Run ```docker build -f Dockerfile -t <username>/<imagename>:<tag> .``` to build the project.
1. Adapt the docker project name in docker_script.sh accordingly.
1. Run ```bash docker_script python code/hatebert_classification.py``` to classify offensive preferred answers in the dataset.
1. Run ```bash docker_script python code/hatebert_filtering.py``` to filter the dataset based on the results of the classification.
