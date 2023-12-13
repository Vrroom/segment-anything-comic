# Segment Anything Comic

Code for training and running a SAM model, modified for predicting polygon segmentations of comic frames.

## Installation

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
