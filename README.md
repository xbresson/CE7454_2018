## CE7454: Deep Learning for Data Science <br> Semester 1 2018/19 <br> Xavier Bresson
    


<br>
<br>

## Running Python notebooks without local Python installation
<br>

&nbsp;&nbsp;&nbsp; Run the notebooks from the cloud using Binder: Simply [click here].

[Click here]: https://mybinder.org/v2/gh/xbresson/CE7454_2018/master




<br>
<br>

## Local Python installation
<br>

Follow the following instructions to install Miniconda and create a Python environment for the course:

1. Download the Python 3.6 installer for Windows, macOS, or Linux from <https://conda.io/miniconda.html> and install with default settings. Note for Windows: If you don't know if your operating system is 32-bit or 64-bit, then open Settings-System-About-System type to find out your xx-bit system.
   * Windows: Double-click on the `Miniconda3-latest-MacOSX-x86_64.exe` file. 
   * macOS: Run `bash Miniconda3-latest-MacOSX-x86_64.sh` in your terminal.
   * Linux: Run `bash Miniconda3-latest-Linux-x86_64.sh` in your terminal.
1. Windows: Open the Anaconda Prompt terminal from the Start menu. MacOS, Linux: Open a terminal.
1. Install git: `conda install git`.
1. Download the GitHub repository of the course: `git clone https://github.com/xbresson/CE7454_2018`.
1. Go to folder CE7454_2018 with `cd CE7454_2018`, and create a Python virtual environment with the packages required for the course: `conda env create -f environment.yml`. Note that the environment installation may take some time. 



   Notes: <br>
      The installed conda packages can be listed with `conda list`.<br>
      Some useful Conda commands are `pwd`, `cd`, `ls -al`, `rm -r -f folder/`<br>
      Add a python library to the Python environment: `conda install -n CE7454_2018 numpy` (for example)<br>
      Read [Conda command lines for packages and environments]<br>
      Read [managing Conda environments]

[managing Conda environments]: conda/conda_environments.pdf

[Conda command lines for packages and environments]: conda/conda_cheatsheet.pdf




<br> 
<br> 

## Running local Python notebooks 
<br>


1. Windows: Open the Anaconda Prompt terminal from the Start menu. MacOS, Linux: Open a terminal.
1. Activate the environment. Windows: `activate CE7454_2018`, macOS, Linux: `source activate CE7454_2018`.
1. Download the python notebooks by direct downloads from the next section or with GitHub with the command `git pull`. 
1. Start Jupyter with `jupyter notebook`. The command opens a new tab in your web browser.
1. Go to the exercise folder, for example `CE7454_2018/codes/lab01_python`.
1. When your tutorial is completed, you can go back to the terminal command by shutting down the juypter kernels with `Control-C`. 

	Notes:<br> 
      Windows: Folder CE7454_2018 is located at `C:\Users\user_name\CE7454_2018`. MacOS, Linux: `/Users/user_name/CE7454_2018`.<br>







[python]: https://www.python.org
[scipy]: https://www.scipy.org
[anaconda]: https://anaconda.org
[miniconda]: https://conda.io/miniconda.html
[conda]: https://conda.io
[conda-forge]: https://conda-forge.org




<br> 
<br> 

## Python notebooks of the course
<br>

[Download datasets]

[Tutorial 1: Introduction to Python]

[Tutorial 2: Introduction to PyTorch]

[Tutorial 3: PyTorch Tensor 1]

[Tutorial 4: PyTorch Tensor 2]

[Download datasets]: codes/zip/data.zip

[Tutorial 1: Introduction to Python]: codes/zip/lab01_python.zip

[Tutorial 2: Introduction to PyTorch]: codes/zip/lab02_pytorch.zip

[Tutorial 3: PyTorch Tensor 1]: codes/zip/lab03_pytorch_tensor1.zip

[Tutorial 4: PyTorch Tensor 2]: codes/zip/lab04_pytorch_tensor2.zip




<br>
<br>
<br>
<br>
<br>
<br>



