## Required Packages

```
brew install pyenv
brew install pyenv-virtualenv
```

## Install CUDA

## Initialize pyenv and pyenv-virtualenv

Add the following to .zshrc
```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```
Reinitialize shell,

```
source .zshrc
```

## Install python

```
pyenv install 3.6.1
```

## Install Xcode Command Line Tools
Markf
If the python build fails because of missing libraries install Xcode command line tools and try again.

```
xcode-select --install
```

## Create Virtual Environment in Project Directory

```
pyenv virtualenv 3.6.1 neural_net
pyenv local 3.6.1
```

## Activate Virtual Environment

```
pyenv activate neural_net
```

## Install Packages

```
pip install -r requirements.txt
```

## Start Editing

If ```atom``` is used start from the command line in the project directory to be sure the virtual environment is recognized during the editing session.


### The MNIST Dataset of Handwritten Numbers

> [Training data](http://www.pjreddie.com/media/files/mnist_train.csv)
>
> [Test data](http://www.pjreddie.com/media/files/mnist_test.csv)
