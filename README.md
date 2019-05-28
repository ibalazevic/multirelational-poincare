
## Multi-relational Poincaré Graph Embeddings


Multi-relational link prediction in the Poincaré ball model of hyperbolic space.

<p align="center">
  <img src="https://raw.githubusercontent.com/ibalazevic/multirelational-poincare/master/murp_vs_mure.png"/ width=700>
</p>


This codebase contains PyTorch implementation of the paper:

> Multi-relational Poincaré Graph Embeddings.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> arXiv preprint arXiv:1905.09791, 2019.
> [[Paper]](https://arxiv.org/pdf/1905.09791.pdf)

### Link Prediction Results

Model | Dataset | dim | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :--- | :---: | :---: | :---: | :---: | :---:
MuRP | WN18RR | 40 |  0.477 | 0.555 | 0.489 | 0.438
MuRP | WN18RR | 200 |  0.481 | 0.566 | 0.495 | 0.440
MuRE | WN18RR | 40 |  0.459 | 0.528 | 0.474 | 0.429
MuRE | WN18RR | 200 |  0.475 | 0.554 | 0.487 | 0.436
MuRP | FB15k-237 | 40| 0.324 | 0.506 | 0.356 | 0.235
MuRP | FB15k-237 | 200| 0.335 | 0.518 | 0.367 | 0.243
MuRE | FB15k-237 | 40| 0.315 | 0.493 | 0.346 | 0.227
MuRE | FB15k-237 | 200| 0.336 | 0.521 | 0.370 | 0.245


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

    @article{balazevic2019multi,
    title={Multi-relational Poincar$\backslash$'e Graph Embeddings},
    author={Bala{\v{z}}evi{\'c}, Ivana and Allen, Carl and Hospedales, Timothy},
    journal={arXiv preprint arXiv:1905.09791},
    year={2019}
    }


