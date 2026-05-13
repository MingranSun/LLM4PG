# LLM4PG
M. Sun, L. Bai, X. Cheng, and J. Wu, “LLM4PG: Adapting Large Language Model for Pathloss Map Generation via Synesthesia of Machines,” IEEE Transactions on Machine Learning in Communications and Networking, accepted, 2026. 

[fig4_v2.pdf](https://github.com/user-attachments/files/27686345/fig4_v2.pdf)


## Dependencies and Installation
- Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU/CPU + CUDA 
- Python packages: `pip install -r requirements.txt`


## Dataset Preparation
The test datasets used in this paper can be downloaded in the following links [[Dataset]](https://pan.baidu.com/s/1PZjVKNHknX85qxxj3Iw25Q?pwd=97tz) with the extraction code 97tz. 

## Get Started
### Step1: Prepare the Files
- Dataset: Download the dataset and place it under the `dataset/` folder in the root directory.
- GPT-2 Files: Download the [[GPT-2]](https://pan.baidu.com/s/1DhLo2zKKKwiKNFuN2XJGqw?pwd=89xb) files with the extraction code 89xb files and put them into the `gpt2/` folder.
- LLM4PG Weights: We have released the model weights for inference in [[Model]](https://pan.baidu.com/s/11Lv-JMNrP8s4oQwC6j-7rg?pwd=25m6) with the extraction code 25m6. Download them and place it under the `weights/` folder.

### Step2: Run Inference
Once all the required files are in place, you can run the following command:
```
python inference.py
```

## Citation
If you find this repo helpful, please cite our paper.
```@article{sun2026llm4pg,
  title={LLM4PG: Adapting Large Language Model for Pathloss Map Generation via Synesthesia of Machines},
  author={Sun, Mingran and Bai, Lu and Cheng, Xiang and Wu, Jianjun},
  journal={IEEE Transactions on Machine Learning in Communications and Networking},
  year={2026}
}
```

