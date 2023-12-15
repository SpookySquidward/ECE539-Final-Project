# ECE539-Final-Project
Predicting the sentiment of Amazon reviews based of their text

### Getting Started
1. Make sure you have installed the following dependencies:
    - [<code>git</code>](https://git-scm.com/downloads) (run <code>git --version</code> from a command line to check, and upgrade if requested)
    - [Python 3.11](https://www.python.org/downloads/release/python-3116/) (run <code>python --version</code> or <code>py --version</code> from a command line to check your current version; you can select your installation when opening the project, too)
    - [VS Code](https://code.visualstudio.com/download) (or use your own IDE, I don't care)
2. Clone this repo to your local machine
    - For VS Code, follow Microsoft's docs [here](https://code.visualstudio.com/docs/sourcecontrol/github); I recommend signing into your GitHub account from VS Code to easily push your changes to the central repo (see the docs [here](https://code.visualstudio.com/docs/sourcecontrol/github))
    - Pick a location with a few GB of free space to clone to, as the data for this project may exceed 10GB.
3. Install required dependencies for this project from <code>requirements.txt</code> via pip
    - I recommend setting up a Python virtual environment to manage packages; this isolates your Python interpreter and package installations so they don't conflict with other versions used for other projects on your system and can be uninstalled easily later. For VS Code, follow [Microsoft's docs](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command) (feel free to ask for help here); VS Code will then use your virtual environment packages automatically whenever you open your code workspace
    - Once you have created a virtual environment with a Python 3.11.x interpreter, run <code>./.venv/Scripts/pip install -r requirements.txt</code> (Windows) to install required packages (<code>numpy</code>, <code>torch</code>, <code>matplotlib</code>, <code>scikit-learn</code>, and their dependencies)

### Proof-of-concept (no downloads required)
1. Run proof-of-concept.py

### Use entire dataset
1. Download the dataset [here](https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view)
    - You'll have to extract the tarball to get to the plain-text <code>.csv</code> files; Windows 11 and Linux can do this natively, or try [7-Zip](https://www.7-zip.org/download.html)
2. Copy the two <code>.csv</code> files and <code>readme.txt</code> from the dataset to the <code>./dataset</code> folder of your clone
3. Run <code>test_install.py</code> to verify that your environment is configured correctly and that you can import data
