# Korean morphological analyzer (한국어 형태소 분석기)

PyTorch implementation for Korean morphological analyzer

### Dependency
- PyTorch >= 1.1
- torchtext
- Check the requirements.txt

```bash
pip install -r requirements.txt
```

## Getting Started

### Step 1: Prepare the data
The sample data can be found data/ directory. The data consists of eojoel and pairs of morphmeme and POS tag.

### Step 2: Train the model
```bash
python train.py 
```
This will load a config file (config/kma.yaml) and run the model defined by the config file, 
which consists of a 3-layer LSTM with 100 hidden units on the bidirectional encoder
and a [Pointer-generator network](https://aclweb.org/anthology/P17-1099) and a CRF tagger.
The detailed parameters can be found config/ directory.

### Step 3: Tagging
```bash
python tagging.py --input_file text_file --output output_file
```
We have a model which you can use to tag on new data. It reads sentences line by line and executes the tagging.
The tagged outputs are saved into output_file.

### Pretrained model
- Pretrained models can be downloaded [download](https://drive.google.com/open?id=192tv3ZYPieFv5ubVgA2y7tdAm2p7ggiR)

## Acknowledgement
The implementation is highly inspired from [IBM's seq2seq](https://github.com/IBM/pytorch-seq2seq)
and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).


