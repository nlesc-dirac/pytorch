# LBFGS optimizer
Only the stochastic LBFGS optimizer is provided with the code. Further details are given [in this paper](https://ieeexplore.ieee.org/document/8755567). Also see [this introduction](http://sagecal.sourceforge.net/pytorch/index.html).

Files included are:

``` lbfgsnew.py ```: New LBFGS optimizer

``` lbfgs.py ```: Symlink to ``` lbfgsnew.py ```


Examples of use:

  * Federated learning: see [these examples](https://github.com/SarodYatawatta/federated-pytorch-test).

  * Calibration and other inverse problems: see [radio interferometric calibration](https://github.com/SarodYatawatta/calibration-pytorch-test).

  * Other problems: see [this example](https://ieeexplore.ieee.org/abstract/document/8588731).
