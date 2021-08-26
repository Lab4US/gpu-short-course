# IUS-2021 GPU short-course

This is a repository for the “Digital Signal Processing with GPUs — Introduction to Parallel Programming” short-course.

Note: To be informed about any changes in the future, you can press the "Star" and "Watch" buttons, which you can find in the upper right corner. Thanks!

## Contents
- `slides`: decks of slides for the presentation,
- `exercises`: Jupyter notebooks with CUDA code examples to run,
  - `cupy`: a list of exercise notebooks that use the CuPy package to communicate with the GPU. **These are the notebooks we use in the IUS 2021 short course video**. Each subdirectory contains the following files and directories:
    - `*.ipynb`: exercise jupyter notebooks,
    - `*.cc`: CUDA C kernel source files,
    - subdirectory `solutions`: solutions to the exercises (only in the case of notebooks with some exercises). 
  - `numba`: a list of exercise notebooks that use the Numba package to communicate with the GPU. The notebooks are from the older edition of the course and may be useful in case you are not familiar with C language (which is required in cupy courses to write CUDA C GPU kernels). Note: we do not describe these notebooks in the IUS 2021 short course video.
- `cfg` - configuration files for your Conda environment and Docker images,
- `utils` - utility Python scripts.

## Organization
The short-course is organized by [us4us Ltd.](http://us4us.eu/), [IPPT PAN](http://www.ippt.pan.pl/en/), and [LITMUS, University of Waterloo](https://lit-mus.org/about/). The pre-recorded lectures are available for IUS 2021 "Ultrasound Signal Processing with GPUs — Introduction to Parallel Programming" short-course participants.

## Jupyter notebooks

In this section we describe three possible options how to get and run the exercise jupyter notebooks.

### Option #1: Docker image

Requirements:
- Linux: make sure that your GPU and operating system are supported by NVIDIA Container Toolkit (check the list available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#linux-distributions))
- Windows: make sure your GPU and OS are supported by CUDA Toolkit on WSL (check requirements [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#wsl2-system-requirements))

Before running the exercise jupyter notebooks please install [docker](https://docs.docker.com/get-docker/) on your computer. To be able to use the GPU as part of the docker container, it is also necessary to install the following software:

- on Linux: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- on Windows: install [NVIDIA CUDA on Windows Subsystem for Linux](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

Then just run the following command in Linux or WSL terminal:

```
sudo docker run -p 8888:8888  -it --gpus all us4useu/ius_gpu_short_course:1.0.0
```

Note: if for some reason you encounter a problem with installing CUDA Container Toolkit or NVIDIA CUDA on WSL (e.g. your GPU is not supported by WSL), consider using the Miniconda environment on your host machine (see instructions below).

### Option #2: running jupyter notebooks in Miniconda

We recommend using [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-11.0-download-archive).

Python 3.8: we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead of using Python distribution available in your operating system. Miniconda gives you a possibility to create an isolated Python environment, with it's own set of software and packages. Any changes you will make in the environment will **not** impact your system-wide configuration.

1. Install Minconda for Python 3.8.
2. Open your shell (Linux) or Anaconda Powershell Prompt (Windows).
3. Create a new conda environment: 
```
conda create -n gpu-course python=3.8
```
7. Activate the environment: 
```
conda activate gpu-course
```
9. Clone this repository on your computer.
```
git clone https://github.com/us4useu/ius-2021-gpu-short-course.git
cd ius-2021-gpu-short-course
```
11. Install requirements. **Note**: if you are using a version of CUDA other than 11.0, be sure to change the version of cuda in the name of cupy package (i.e. the cupy-cuda110 to cupy-cudaXY, where X.Y is the version of CUDA you currently use).
```
conda env update --name gpu-course --file cfg/conda-requirements.yml --prune 
pip install -r cfg/pip-requirements.txt
pip install cupy-cuda110==9.3.0
pip install .
```
13. Run: `jupyter lab`
14. Open one of the exercise notebooks and run all cells to test if everything works correctly.


### Option #3: running the jupyter notebooks on Google Colab

It's also possible to run via Google Colab website (see instructions above) instead of using Jupyter Lab on your computer. 

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
