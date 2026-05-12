# LLM4PG
M. Sun, L. Bai, X. Cheng, and J. Wu, “LLM4PG: Adapting Large Language Model for Pathloss Map Generation via Synesthesia of Machines,” IEEE Transactions on Machine Learning in Communications and Networking, accepted, 2026. 

## Dependencies and Installation
- Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU/CPU + CUDA 
- Python packages: `pip install -r requirements.txt`


## Get Started
We have released the model weight for inference in [[Model]](https://pan.baidu.com/s/1ewhFuIBm3os6L1n-tviiNw?pwd=p3a3) with the extraction code p3a3.


### Inference command 
```
python test_image_pl.py
```

## Citation
If you find this repo helpful, please cite our paper.
```latex
@ARTICLE{10614105,
  author={Sun, Mingran and Bai, Lu and Huang, Ziwei and Cheng, Xiang},
  journal={IEEE Wireless Communications Letters}, 
  title={Multi-Modal Sensing Data-Based Real-Time Path Loss Prediction for 6G UAV-to-Ground Communications}, 
  year={2024},
  volume={13},
  number={9},
  pages={2462-2466},
  keywords={Sensors;Wireless sensor networks;Wireless communication;Autonomous aerial vehicles;Real-time systems;6G mobile communication;Loss measurement;6G UAV-to-ground communications;path loss prediction;sensing and communication integration},
  doi={10.1109/LWC.2024.3419245}}
```
