# IUS GPU short-course

This is a repository for the “Ultrasound/Digital Signal Processing with GPUs — Introduction to Parallel Programming” short-course.
The course overview is [here (PDF)](slides/US-GPU-short-course-info.pdf).

**The course has been originally prepared and presented at the IEEE International Ultrasonic Symposium (IUS) in 2021 Next, it has been revised and extended for the IUS-2022 and IUS-2023!**

*Note: To be informed about any changes in the future, you can press the "Star" and "Watch" buttons, which you can find in the upper right corner. Thanks!*

## Additional/Specific Info 
* [Short-course @ IUS-2023 (1 SEP 2023)](2023-IUS-US-GPU.md)
  * [CUDA refresher](https://github.com/Lab4US/gpu-short-course/blob/main/slides/ius-2023/IUS-2023-US-GPU-CUDA-refresher.pdf)
  * [CUDA by example: delay and sum](https://github.com/Lab4US/gpu-short-course/blob/main/slides/ius-2023/cuda_by_example_delay_and_sum.pdf) 
  * [Real-Time Speed of Sound Estimation and CNN Inferencing Using GPU](https://github.com/Lab4US/gpu-short-course/blob/main/slides/ius-2023/IUS-2023-US-GPU.pdf) 
* [Short-course @ TIB PAN (May 2023)](2023-TIB-DSP-GPU.md)
* Short-course @ [IUS-2022 (October 2022)](https://2022.ieee-ius.org/short-courses/)
* Short-course @ [IUS-2021 (September 2021)](https://2021.ieee-ius.org/short-courses/)
* Short-course @ [Biocentrum Ochota (June 2021)](http://www.biocentrumochota.pan.pl/)

<hr>

## Contents
- `slides`: decks of slides for the presentation,
- `exercises`: Jupyter notebooks with CUDA code examples to run,
  - `cupy`: a list of exercise notebooks that use the CuPy package to communicate with the GPU. **These are the notebooks we use in the IUS short course video**. Each subdirectory contains the following files and directories:
    - `*.ipynb`: exercise jupyter notebooks,
    - `*.cc`: CUDA C kernel source files,
    - subdirectory `solutions`: solutions to the exercises (only in the case of notebooks with some exercises). 
  - `numba`: a list of exercise notebooks that use the Numba package to communicate with the GPU. The notebooks are from the older edition of the course and may be useful in case you are not familiar with C language (which is required in cupy courses to write CUDA C GPU kernels). Note: we do not describe these notebooks in the IUS short course video.
- `cfg` - configuration files for your Conda environment and Docker images,
- `utils` - utility Python scripts,
- Tensorflow2TensorRTDemo_IUS2023 -- Tensorflow to ONNX format example, with TensorRT processing, IUS 2023.  

### Video
Video recordings of the lectures, exercises, and case-studies are available on-line:
- [Vimeo playlist](https://vimeo.com/showcase/2022-us-gpu-short-course)
- [YouTube playlist](https://www.youtube.com/playlist?list=PLTXwDWOjJ0Xeisir2sL3RxkC1RHpMmFbG)

*Note: slides have been updated for 2022, while most of the recordings are from 2021.*


## Organization
The short-course is organized by [us4us Ltd.](http://us4us.eu/), [IPPT PAN](http://www.ippt.pan.pl/en/), and [LITMUS, University of Waterloo](https://lit-mus.org/about/). The pre-recorded lectures are available for IUS "Ultrasound Signal Processing with GPUs — Introduction to Parallel Programming" short-course participants.

All the IUS "Ultrasound Signal Processing with GPUs — Introduction to Parallel Programming" short-course exercise recordings were performed on:
- Ubuntu 18.04 LTS,
- NVIDIA Titan X GPU,
- exercise notebooks version 1.0.

We recommend the use of notebooks in the latest version: they may differ slightly from those presented in exercise recordings, but also may contain improvements to version 1.0.

## Exercise notebooks

In this section we describe options how to get and run the exercise Jupyter notebooks.

### Option #1: running jupyter notebooks in Miniconda (Linux x64 or Windows x64)

We recommend using [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-11.0-download-archive).

Python 3.8: we recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instead of using Python distribution available in your operating system. Miniconda gives you a possibility to create an isolated Python environment, with it's own set of software and packages. Any changes you will make in the environment will **not** impact your system-wide configuration.

1. Install Minconda for Python 3.8.
2. Install `wget` program:
  - Linux: use your package manager to get the app, for example on Ubuntu: `sud apt install wget`
  - Windows: download the following [executable](https://eternallybored.org/misc/wget/1.21.1/64/wget.exe) and put it into a directory listed in your `PATH` environment variable. 
3. Open your shell (Linux) or Anaconda Powershell Prompt (Windows).
4. Create a new conda environment: 
```
conda create -n gpu-course python=3.8
```
5. Activate the environment: 
```
conda activate gpu-course
```
6. Clone this repository on your computer.
```
conda install git
git clone https://github.com/Lab4US/gpu-short-course.git --branch v2.0 --single-branch
cd gpu-short-course
```
7. Install in your conda environment the required software:
  - Linux: `conda env update --name gpu-course --file cfg/conda-requirements-linux.yml --prune` 
  - Windows: `conda env update --name gpu-course --file cfg/conda-requirements-windows.yml --prune` 
8. Install the required Python packages. **Note**: if you are using a version of CUDA other than 11.0, be sure to change the version of cuda in the name of cupy package (i.e. the cupy-cuda110 to cupy-cudaXY, where X.Y is the version of CUDA you currently use).
```
pip install -r cfg/pip-requirements.txt
pip install cupy-cuda110==9.3.0
pip install -e ./utils 
```
9. Run: `jupyter lab`

You should see an output similar to the one below:
```
[I 2021-08-26 18:28:10.973 ServerApp] jupyterlab | extension was successfully linked.
[W 2021-08-26 18:28:10.990 ServerApp] The 'min_open_files_limit' trait of a ServerApp instance expected an int, not the NoneType None.
[I 2021-08-26 18:28:11.255 ServerApp] nbclassic | extension was successfully loaded.
[I 2021-08-26 18:28:11.256 LabApp] JupyterLab extension loaded from C:\Users\username\anaconda3\envs\gpu-course\lib\site-packages\jupyterlab
[I 2021-08-26 18:28:11.257 LabApp] JupyterLab application directory is C:\Users\username\anaconda3\envs\gpu-course\share\jupyter\lab
[I 2021-08-26 18:28:11.261 ServerApp] jupyterlab | extension was successfully loaded.
[I 2021-08-26 18:28:11.262 ServerApp] Serving notebooks from local directory: C:\Users\username\repos\gpu-short-course
[I 2021-08-26 18:28:11.262 ServerApp] Jupyter Server 1.10.2 is running at:
[I 2021-08-26 18:28:11.263 ServerApp] http://localhost:8888/lab?token=ff2ac4b0004ce179455a5e48b24defd541a0869aee7fe33d
[I 2021-08-26 18:28:11.263 ServerApp]  or http://127.0.0.1:8888/lab?token=ff2ac4b0004ce179455a5e48b24defd541a0869aee7fe33d
[I 2021-08-26 18:28:11.265 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2021-08-26 18:28:11.353 ServerApp]

    To access the server, open this file in a browser:
        file:///C:/Users/username/AppData/Roaming/jupyter/runtime/jpserver-24164-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=ff2ac4b0004ce179455a5e48b24defd541a0869aee7fe33d
     or http://127.0.0.1:8888/lab?token=ff2ac4b0004ce179455a5e48b24defd541a0869aee7fe33d
[I 2021-08-26 18:28:14.352 LabApp] Build is up to date
```

10. Open one of the exercise notebooks with solutions and run all cells to test if everything works correctly.

### Option #2: Docker image (Linux x64 only)

Requirements:
- Linux: make sure that your GPU and operating system are supported by NVIDIA Container Toolkit (check the list available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#linux-distributions))

Before running the exercise jupyter notebooks please install [docker](https://docs.docker.com/get-docker/) on your computer. To be able to use the GPU as part of the docker container, it is also necessary to install the following software:

- on Linux: install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Then just run the following command in Linux:
``` 
sudo docker run -p 8888:8888  -it --gpus all --name gpu_course us4useu/ius_gpu_short_course:2.0
```

You should see an output similar to the one below:
```
Unable to find image 'us4useu/ius_gpu_short_course:2.0' locally
1.1: Pulling from us4useu/ius_gpu_short_course
16ec32c2132b: Already exists
ab49a37cda04: Already exists
b23b1cc2f66c: Already exists
de57da913f8f: Already exists
06b8d7b81090: Already exists
5e0c69981f75: Already exists
aa4d68208e9a: Already exists
cdfffea224b0: Already exists
6aea04a7cdf1: Already exists
41e54634e42b: Already exists
f2afcbd4e723: Pull complete
09f295b33740: Pull complete
8c6ebaa88ded: Pull complete
a5ada5e02db8: Pull complete
3e6037c11955: Pull complete
926d9389a815: Pull complete
a8484209f71f: Pull complete
6fb4d747b5b4: Pull complete
9547bfd6657a: Pull complete
6eadc574abfe: Pull complete
7a9ec0c5114e: Pull complete
6e50ce083ad1: Pull complete
08d362a2355f: Pull complete
88aa9ffa018d: Pull complete
5fac73b69e4e: Pull complete
bb3b949600c9: Pull complete
Digest: sha256:14dd7ea7c3a8943b88bd937fe3e52741fe51fec282822d62d16b00765effbd00
Status: Downloaded newer image for us4useu/ius_gpu_short_course:2.0
Running as student
Executing the command: jupyter lab --no-browser
[I 2021-08-26 11:10:19.003 ServerApp] jupyterlab | extension was successfully linked.
[I 2021-08-26 11:10:19.019 ServerApp] Writing Jupyter server cookie secret to /home/student/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2021-08-26 11:10:19.217 ServerApp] nbclassic | extension was successfully linked.
[I 2021-08-26 11:10:19.275 ServerApp] nbclassic | extension was successfully loaded.
[I 2021-08-26 11:10:19.277 LabApp] JupyterLab extension loaded from /opt/conda/lib/python3.8/site-packages/jupyterlab
[I 2021-08-26 11:10:19.277 LabApp] JupyterLab application directory is /opt/conda/share/jupyter/lab
[I 2021-08-26 11:10:19.281 ServerApp] jupyterlab | extension was successfully loaded.
[I 2021-08-26 11:10:19.282 ServerApp] Serving notebooks from local directory: /home/student/gpu-short-course
[I 2021-08-26 11:10:19.282 ServerApp] Jupyter Server 1.10.2 is running at:
[I 2021-08-26 11:10:19.282 ServerApp] http://47ebf762524f:8888/lab?token=1f048e5cfd1b9bd592885ab6c599e3c7e6c62a7168a7d7ce
[I 2021-08-26 11:10:19.282 ServerApp]  or http://127.0.0.1:8888/lab?token=1f048e5cfd1b9bd592885ab6c599e3c7e6c62a7168a7d7ce
[I 2021-08-26 11:10:19.282 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2021-08-26 11:10:19.287 ServerApp]
    To access the server, open this file in a browser:
        file:///home/student/.local/share/jupyter/runtime/jpserver-7-open.html
    Or copy and paste one of these URLs:
        http://47ebf762524f:8888/lab?token=1f048e5cfd1b9bd592885ab6c599e3c7e6c62a7168a7d7ce
     or http://127.0.0.1:8888/lab?token=1f048e5cfd1b9bd592885ab6c599e3c7e6c62a7168a7d7ce
```

Now, copy and paste the Jupyter Lab URL (`http://127.0.0.1:8888/lab?token=1f048e5cfd1b9bd592885ab6c599e3c7e6c62a7168a7d7ce` in our case) to your web browser. 

To stop the container: just press CTRL + C, or run `docker stop gpu_course`. To start the container again, use `docker start -i gpu_course`.

To access docker container data (e.g. NVVP report results), you can use `docker cp` command, for example:

```
sudo docker cp gpu_course:/home/student/gpu-short-course/exercises/cupy/1_CUDA_programming_model/solutions/nvvp_example.nvvp .
```

## Team
The course is presented by Dr Marcin Lewandowski, Piotr Jarosik and Dr Billy Yiu.

Course support team includes: Ziemowit Klimonda, Mateusz Walczak, Piotr Karwat and Julia Lewandowska.

*This short-course is a part of the [Lab4US.eu](https://lab4us.eu) initiative*
[![Lab4US](figs/Lab4US-banner-EN-800.png)](https://lab4us.eu)

## License
Materials for the short-course „Digital Signal Processing with GPUs — Introduction to Parallel Programming” are licensed by us4us Ltd. under the Creative Commons Attribution-NonCommercial 4.0 International License.
Some slides and examples are borrowed from the course „The GPU Teaching Kit” that is licensed by NVIDIA and the University of Illinois under the Creative Commons Attribution-NonCommercial 4.0 International License.

![CC BY NC](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png "CC BY NC")
