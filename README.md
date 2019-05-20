
## Multi-relational Poincaré Graph Embeddings

This codebase contains PyTorch implementation of the paper:

> Multi-relational Poincaré Graph Embeddings.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> arxiv link: TBD

### Link Prediction Results

Dataset | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.358 | 0.544 | 0.394 | 0.266
WN18RR | 0.470 | 0.526 | 0.482 | 0.443

### Running a model

To run the model, execute the following command:

     CUDA_VISIBLE_DEVICES=0 python main.py --model poincare --dataset WN18RR --num_iterations 500 
                                           --nneg 50 --batch_size 128 --lr 50 --dim 40 

Available datasets are:
    
    FB15k-237
    WN18RR
    
To reproduce the results from the paper, use learning rate 50 for WN18RR and learning rate 10 for FB15k-237.


### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1
    
### Citation

If you found this codebase useful, please cite:

TBD

