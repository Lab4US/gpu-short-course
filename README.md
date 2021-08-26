# BIOCENTRUM / IUS-2021 GPU short-course

This is a repository for the “Digital Signal Processing with GPUs — Introduction to Parallel Programming” short-course.

Note: this course is still a work-in-progress effort, so in the future we will supplement it with new materials and examples. To be informed about any changes in the future, you can press the "Star" and "Watch" buttons, which you can find in the upper right corner. Thanks!

## Contents
- slides — decks of slides for the presentation; 
- exercises — Jupyter notebooks with CUDA code examples to run,
- cfg - configuration files 

## Organization
The short-course is organized by [us4us Ltd.](http://us4us.eu/) and [IPPT PAN](http://www.ippt.pan.pl/en/). pre-recorded on [IEEE IUS-2021](https://2021.ieee-ius.org/short-courses/).

In both cases the links will be provided by email to the registered participants.

## Jupyter notebooks

In this section we describe how you can get and run the exercise jupyter notebooks.

### Using ius_gpu_short_course Docker image

We recommend using `us4useu/ius_gpu_short_course:1.0.0` Docker image.

Before running the exercise jupyter notebooks please install [docker](https://docs.docker.com/get-docker/) on your computer. To be able to use the GPU as part of the docker container, it is necessary to install the following additional software:

- on Linux: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - make sure that your operating system is supported by NVIDIA Container Toolkit (check the list available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#linux-distributions))
- on Windows: install [NVIDIA CUDA on Windows Subsystem for Linux](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  - make sure your GPU and OS is supported by CUDA Toolkit on WSL: [requirements](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#wsl2-system-requirements)

Then just run the following command in Linux or WSL terminal:

```
sudo docker run -p 8888:8888  -it --gpus all us4useu/ius_gpu_short_course:1.0.0
```

Note: if for some reason you encounter a problem with the installation of CUDA on Linux or for WSL (e.g. your GPU is not supported by WSL), another option might be to use the Miniconda environment on your host machine (see instructions below).

### Running jupyter notebooks in Miniconda

Install the following software first:
- we recommend using [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-11.0-download-archive),
- Python 3.8: we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead of using Python distribution available in your operating system. Miniconda gives you a possibility to create an isolated Python environment, with it's own set of software and packages. Any changes you will make in the environment will **not** impact your system-wide configuration.
  1. Install Minconda for Python 3.8.
  3. Open your shell (Linux or MacOS) or Anaconda Powershell Prompt (Windows).
  4. Create a new environment: `conda create -n gpu-course python=3.8`
  5. Activate the environment: `conda activate gpu-course`
  6. Clone this repository on your computer.
  7. Install requirements: `conda env update --name gpu-course --file cfg/conda-env-linux.yml --prune && pip install cupy-cuda110==9.3.0 && pip install ` 
  9. Run: `jupyter lab`
  10. Open one of the exercise notebooks and run all cells.

It's also possible to run via Google Colab website (see instructions above) instead of using Jupyter Lab on your computer. 

### Running the jupyter notebooks on Google Colab

If you don't have an access to NVIDIA GPU card, you can try running the notebooks on Google Colab.

Copy the notebooks from this repository to some location on your Google Drive. 

Please remember to change runtime to GPU before running notebooks (e.g. see instructions [here](https://www.geeksforgeeks.org/how-to-use-google-colab/)). 

Caution:

- Google Colab in the free version does not guarantee the availability of the GPU in any particular time.
- Google Colab in the free version may provide one of the following graphics cards: NVIDIA T4 or NVIDIA K80. NVIDIA T4 may not work with NVPROF profiler. In the future, we will supplement the course with examples for the new profiler required by the latest graphics cards: NVIDIA Nsight Compute and Systems.

Still, it is possible to run Google Colab notebook on on your computer's GPU - you can change the runtime to your local jupyter instance. Install the required software on your computer first (see instruction: *Running the jupyter notebooks on your computer*), then follow the Google Colab [instructions](https://research.google.com/colaboratory/local-runtimes.html).

## Team
The course is presented by Dr Marcin Lewandowski, Piotr Jarosik and Billy Yiu.

Course support team includes: Mateusz Walczak, Piotr Karwat, Ziemowit Klimonda and Julia Lewandowska.

## License
Materials for the short-course „Digital Signal Processing with GPUs — Introduction to Parallel Programming” are licensed by us4us Ltd. the IPPT PAN under the Creative Commons Attribution-NonCommercial 4.0 International License.
Some slides and examples are borrowed from the course „The GPU Teaching Kit” that is licensed by NVIDIA and the University of Illinois under the Creative Commons Attribution-NonCommercial 4.0 International License.

![CC BY NC](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png "CC BY NC")
