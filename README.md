# Segment Anything Comic

Code for training and running a SAM model, modified for predicting polygon segmentations of comic frames. Please see the report `CS_580_Project.pdf` for details.

## Installation

Please ensure you have `conda` installed and be prepared with the full path of your installation for which you'll be prompted. In my case, when I tested it on an A10 GB on Lambda Labs, the path to my `conda` installation was `/home/ubuntu/miniconda3`.

```
git clone https://github.com/Vrroom/segment-anything-comic.git
cd segment-anything-comic
bash install_and_run.sh
```

This will download the model, install dependencies and launch an app that you can visit on localhost to test the model. 

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citations

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
