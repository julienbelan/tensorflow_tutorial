# tensorflow_tutorial
Tutorial to Master the TensorFlow Library

**These beginner tutorials take into account you're able to:**
1. Open a zip file
2. Download software on your computer and follow instructions
3. Run stuff in Terminal (Mac, Linux) or Command Line (Windows)

# Environment: how to set up your computer

You'll need:
- Anaconda: Library with Python + all the stuff you'll need
- Tensorflow: Library for machine learning
- Keras: TensorFlow simplified library for machine learning
- Pycharm: A tool to write Python easily

**Anaconda**
1. Go to [Anaconda Installation Website](https://www.continuum.io/downloads)
2. Just click on the icon that represent your computer.
3. Click on *Python 2.7 graphical installer*
4. Open the download package (.pkg) and follow the installation instruction making sure to install to the default location, note the location (i.e. this/is/a/location).
5. Now you have the Anaconda application installed with Python 2.7, conda package manager and other cool stuff

**TensorFlow**
1. For Mac and Linux: Open Terminal and type *which pip* / For Windows: Open Command Line and type *where pip*
2. Make sure that pip is under the installed anaconda location (the one previously noted)
3. Type *pip install tensorflow*
4. Follow the installation instructions

**Keras**
1. Type *pip install Keras* 

**GitHub**
*GitHub is where you are standing now, it's a shared place to get the tutorials, ask questions or report issues.*
1. Choose the repository tensorflow_tutorial
2. Click on download *zip* on the right corner
3. Open the zip file

**Jupyter**
*Jupyter is a neat way to present python code in a text-like way*
1. Open the Anaconda application named Anaconda-Navigator
2. Open Jupyter Notebook, that'll trigger your browser and open a folder page
3. Go to the tensorflow_tutorial folder
4. Click on the tutorial you wish to do. Notebook are .pynb extensions.

**Pycharm**
*Pycharm is a Python IDE (Integrated Development Environment) it's a all-in-the-box tool for writing Python that allow you to debug and read code easily.*
1. Go to [JetBrain Pycharm](https://www.jetbrains.com/pycharm/download/#section=mac) installation page
2. Choose the good operating system (Windows, macOS or Linux)
3. Choose Community version, odds are that you will never need anything more
4. Open the Pycharm and choose the appearance you wish to use (I find that in general darker is better)
5. Click on *Create New Project*
6. In *Location* pick tensorflow_tutorial folder (remember to use the complete path (aka absolute path))
7. In *Interpreter*, pick the anaconda interpreter we've just installed (that should be something like ~/anaconda2/bin/python)
8. A new window will open, go to *File*, pick *New*, scroll and click on *Python File*

**Now you can just follow the tutorials but keep it mind:**
1. You'll learn by doing so rewrite the tutorials as you go by clicking *new* then *notebook* on the right corner  
2. If things get fuzzy or you happen to run into a bug pull an issue or contact me
3. There exists more modern way to setup an environment (Docker Image, Floyd Machine, etc.) but since most people still do it that way, its good to handle a little bit of it.
