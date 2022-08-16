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
- edge.txt: two rows for each data. Both rows are node indices that one edge connects. The length of the row corresponds to the number of edge.
```
Data0 0 0 0 ...
Data0 1 2 6 ...
```
- node_features.txt: The number of rows for each data depends on the input dimension. E.g, if the input is [x, y, z],then there are 3 rows for each data, one for x, one for y, the other for z.
```
Data0 0.0 0.125 ...
Data0 0.0 0.0 ...
Data0 0.0 0.125 ...
```
- node_labels.txt: Currently we only train on 1D node label.
```
Data0 100050 560992 ...
```

**Train and test**
- Put the three data files named as "edge.txt", "node_features.txt" and "node_labels.txt" under "./data" folder. (You can also specify your own data path using --data_path argument)
- Check optional arguments for training
```
python PNA.py -h
```
- Training (Use Polygraphene dataset as an example): we suggest using multiple GPUs given the high memory requirement.
```
python PNA.py --data_path "./data/" --batch_size 16 --input_dim 3 --num_layer 6 --max_degree 9
```
- Check optional arguments for testing
```
python test.py -h
```
- Testing (Use same arguments as training):
```
python test.py --data_path "./data/" --batch_size 16 --input_dim 3 --num_layer 6 --max_degree 9 --ckpt_path "./pretrained/Graphene/Poly/ckpt/pretrained.pt"
```
**Pretrained models**
- Pretrained models are saved as .pt files corresponding to the three datasets in the folder "pretrained" (models are trained on 4 V100 GPUs). 
- "Graphene/poly" for von Mises stress field prediction in polycrystalline graphene; "Graphene/Porous" for tensile stress (sxx) field prediction in porous graphene membrane; "Al/Poly" for potential energy distribution prediction in polycrystalline aluminum. 
- The details of architectures of pretrained models can be found in the paper.                                  

