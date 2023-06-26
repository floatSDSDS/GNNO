
# GNNO

Public implementation for [WWW'23 paper Neighborhood-based Hard Negative Mining for Sequential Recommendation](https://arxiv.org/pdf/2306.10047.pdf). Largely built on [ReChorus](https://github.com/THUwangcy/ReChorus). 


## Data
We have already included the data folder in our code, if you want to try other datasets, (optional) Run jupyter notebook in data folder to download and build new datasets, or prepare your own datasets according to [Guideline](https://github.com/THUwangcy/ReChorus/blob/master/data/README.md) in `./data`.

## Dependencies

```bash
conda env create -f gcl.yml
```

## Get started
```bash
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
```

## Citation

**If you find this repo is helpful to your research, please cite our paper and either of the Rechorus papers. Thanks!**

```
@article{fan2023neighborhood,
  title={Neighborhood-based Hard Negative Mining for Sequential Recommendation},
  author={Fan, Lu and Pu, Jiashu and Zhang, Rongsheng and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2306.10047},
  year={2023}
}

@inproceedings{wang2020make,
  title={Make it a chorus: knowledge-and time-aware item modeling for sequential recommendation},
  author={Wang, Chenyang and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={109--118},
  year={2020}
}
@article{王晨阳2021rechorus,
  title={ReChorus: 一个综合, 高效, 易扩展的轻量级推荐算法框架},
  author={王晨阳 and 任一 and 马为之 and 张敏 and 刘奕群 and 马少平},
  journal={软件学报},
  volume={33},
  number={4},
  pages={0--0},
  year={2021}
}
```