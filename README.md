## PyTorch BERT 文本分类

### 预训练模型下载路径

```text
https://huggingface.co/bert-base-chinese/tree/main
```
下载config.json, pytorch_model.bin, vocab.txt, 存放在pretrained/bert-base-chinese/文件夹中
```text
pretrained
│  
└─bert-base-chinese
        config.json
        pytorch_model.bin
        vocab.txt

```

**环境**
```text
torch==1.10.1+cu113
transformers==4.15.0
numpy==1.22.3
sklearn==0.0
scikit-learn==1.0.2
tqdm==4.62.3
```

**使用方法**

1. 安装环境
```shell
pip install requirements.txt
```
2. 运行代码
```shell
python main.py
```
### 欢迎star