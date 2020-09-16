# LBFGS optimizer
Only the stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

Examples of use:

  * Federated learning: see [these examples](https://github.com/SarodYatawatta/federated-pytorch-test).

  * Calibration and other inverse problems: see [radio interferometric calibration](https://github.com/SarodYatawatta/calibration-pytorch-test).

  * Other problems: see [this example](https://ieeexplore.ieee.org/abstract/document/8588731).

Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` lbfgs.py ```: Symlink to ``` lbfgsnew.py ```

``` cifar10_resnet.py ```: CIFAR10 ResNet training example (see figures below)

<img src="loss.png" alt="ResNet18/101 training loss/time" width="800"/>

The above figure shows the training loss and training time [using Colab](https://colab.research.google.com/notebooks/intro.ipynb) with one GPU. ResNet18 and ResNet101 models are used. Test accuracy after 20 epochs: 84% for LBFGS and 82% for Adam.

Changing the activation from commonly used ```ReLU``` to others like ```ELU``` gives faster convergence in LBFGS, as seen in the figure below.

<img src="activation.png" alt="ResNet Wide 50-2 training loss" width="400"/>

Here is a comparison of both training error and test accuracy for ResNet9 using LBFGS and Adam.

<img src="resnet9.png" alt="ResNet 9 training loss and test accuracy" width="400"/>

Example usage in full batch mode:

```
from lbfgsnew import LBFGSNew
optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=100, line_search_fn=True, batch_mode=False)
```

Example usage in minibatch mode:

```
from lbfgsnew import LBFGSNew
optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=2, line_search_fn=True, batch_mode=True)
```
