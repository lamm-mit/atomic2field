# Atomic2Field
Codes for translating structural defects to atomic properties 
Z. Yang, M.J. Buehler, Linking Atomic Structural Defects to Mesoscale Properties in Crystalline Solids using Graph Neural Networks, npj Computational Materials, in review

![Overall workflow](https://github.com/lamm-mit/atomic2field/blob/main/IMAGE_github.png)

**Requirements**
```
pip install -r requirements.txt
```

**Dataset**
- Three datasets are given: Al/Poly, Graphene/Poly, Graphene/Porous
- The datasets can be found in the following link: https://www.dropbox.com/sh/w3b8u0i63r2y1kq/AACF8mukZ9nDdG4MGj3F1kCUa?dl=0

**Custom dataset**
- There are 3 files needed to form a dataset: edge.txt, node_features.txt, node_labels.txt. Formats are shown below.
- edge.txt (two rows for each data which correspond to the two nodes that one edge connects)
```
Data0 0 0 0 ...
Data0 1 2 6 ...
```

**Train and test**
- Check optional arguments for training
```
python PNA.py -h
```
- Training (Use Polygraphene dataset as an example): 
```
python PNA.py --batch_size
```
- Testing:
```
python test.py
```
**Pretrained models**
- Pretrained models are saved as .pt files corresponding to the three datasets in the folder "pretrained"                                                        

