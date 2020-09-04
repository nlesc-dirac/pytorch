# LBFGS optimizer
Only the stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

Examples of use:

  * Federated learning: see [these examples](https://github.com/SarodYatawatta/federated-pytorch-test).

  * Calibration and other inverse problems: see [radio interferometric calibration](https://github.com/SarodYatawatta/calibration-pytorch-test).

  * Other problems: see [this example](https://ieeexplore.ieee.org/abstract/document/8588731).

Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` lbfgs.py ```: Symlink to ``` lbfgsnew.py ```

``` cifar10_resnet.py ```: CIFAR10 ResNet training example (see figure below)

<img src="loss.png" alt="ResNet18/101 training loss/time" width="800"/>

The above figure shows the training loss and training time [using Colab](https://colab.research.google.com/notebooks/intro.ipynb) with one GPU. ResNet18 and ResNet101 models are used. Test accuracy after 20 epochs: 84% for LBFGS and 82% for Adam.
