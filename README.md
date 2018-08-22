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
1. Activate the environment. Windows: `activate deeplearn_course`, macOS, Linux: `source activate deeplearn_course`.
1. Download the python notebooks by direct downloads from the next section or with GitHub with the command `git pull`. 
1. Start Jupyter with `jupyter notebook`. The command opens a new tab in your web browser.
1. Go to the exercise folder, for example `CE7454_2018/codes/lab01_python`.


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

Note: The datasets are too large for GitHub. You need to run the scripts in `CE7454_2018/codes/data` or from the zip file [Download datasets] to download the datasets.<br><br>








[Download datasets]

[Download datasets]: codes/zip/data.zip

[Tutorial 1: Introduction to Python]

[Tutorial 1: Introduction to Python]: codes/zip/lab01_python.zip

[Tutorial 2: Introduction to PyTorch]

[Tutorial 2: Introduction to PyTorch]: codes/zip/lab02_pytorch.zip

[Tutorial 3: PyTorch Tensor 1]

[Tutorial 3: PyTorch Tensor 1]: codes/zip/lab03_pytorch_tensor1.zip

[Tutorial 4: PyTorch Tensor 2]

[Tutorial 4: PyTorch Tensor 2]: codes/zip/lab04_pytorch_tensor2.zip

[Tutorial 5: Linear Module]

[Tutorial 5: Linear Module]: codes/zip/lab05_linear_module.zip

[Tutorial 6: Softmax]

[Tutorial 6: Softmax]: codes/zip/lab06_softmax.zip

[Tutorial 7: Vanilla NN]

[Tutorial 7: Vanilla NN]: codes/zip/lab07_vanilla_nn.zip

[Tutorial 8: Train Vanilla NN]

[Tutorial 8: Train Vanilla NN]: codes/zip/lab08_train_vanilla_nn.zip

[Tutorial 9: Minibatch Training]

[Tutorial 9: Minibatch Training]: codes/zip/lab09_minibatch_training.zip

[Tutorial 10: Cross Entropy]

[Tutorial 10: Cross Entropy]: codes/zip/lab10_cross_entropy.zip




<br>
<br>
<br>
<br>
<br>
<br>



