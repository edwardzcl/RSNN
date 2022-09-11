# Residual Spiking Neural Network on a Programmable Neuromorphic Hardware for Speech Keyword Spotting

***
**This code can be used as the supplemental material for the paper: "Residual Spiking Neural Network on a Programmable Neuromorphic Hardware for Speech Keyword Spotting". (Published on *IEEE ICSICT*, October, 2022)**.
***

## Citation:
C. Zou, X. Cui, S. Feng, G. Chen, X. Wang and Y. Wang, "Residual Spiking Neural Network on a Programmable Neuromorphic Hardware for Speech Keyword Spotting," 2022 IEEE 16th International Conference on Solid-State and Integrated CIrcuit Technology (ICSICT), 2022, pp. 1-3, doi: xxx.

### **Features**:
- This supplemental material gives a reproduction function of training and testing experiments of the improved residual RNN (R-SNN) in our paper. Totally, three kinds of optional residual architectures are compared.


## File overview:
- `README.md` - this readme file.<br>
- `data` - the speech dataset folder.<br>
- `figs` - the visualized figure folder.<br>
- `tensorlayer` - the provided binary/ternary package, named [tensorlayer](https://github.com/tensorlayer).<br>
- `tools` - some available scripts.<br>
- `k0B2_asr_shortcut_group4_noplace_A.py` - the training script for the traditional residual architecture i.e. `A` in our paper.<br>
- `k0B2_asr_shortcut_group4_noplace_B.py` - the training script for the traditional residual architecture i.e. `B` in our paper.<br>
- `k0B2_asr_shortcut_group4_noplace_C.py` - the training script for the improve residual architecture in our paper.<br>
- `Spiking_asr_shortcut.py` - the residual SNN inference script.<br>
- `spiking_ulils.py` - the tool script for various spiking operators.<br>

### Requirements:<br>
1. Python-3.6, librosa-0.4<br>
2. Tensorflow 1.2 for cpu or gpu<br>
3. CPU or GPU server

### Usage:
- Please note you have installed the package Tensorflow=1.2.x, then directly run with:

```sh
$ python k0B2_asr_shortcut_group4_noplace_C.py.py --k 0 --B 2 --learning_rate 0.01 --resume False --mode 'training'
```
for the improve residual architecture training,
or
```sh
$ python Spiking_asr_shortcut.py --k 0 --B 2 --noise_ratio 0 --learning_rate 0.01 --resume True --mode 'testing'
```
for the improve residual architecture testing.


## Results
- Please refer to our paper for more information.

## More question:<br>
- There might be a little difference of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: 1801111301@pku.edu.cn if you have any questions or difficulties. I'm happy to help guide you.

### Reference
- [Ternary Weight Networks](http://arxiv.org/abs/1605.04711)
- [Tensorlayer](https://github.com/tensorlayer/TensorLayer)
- [Scatter-and-gather scheme](https://www.frontiersin.org/articles/10.3389/fnins.2021.694170/full)





