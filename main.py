import numpy as np
import torch
import time
from collections import defaultdict
from load_data import Data
from model import *
from rsgd import *
import argparse

    
class Experiment:

    def __init__(self, learning_rate=50, dim=40, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, cuda=False):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_s = model.forward(e1_idx.repeat(len(d.entities)), 
                            r_idx.repeat(len(d.entities)), range(len(d.entities)))

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs==e2_idx.item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
            
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))


    def train_and_eval(self):
        print("Training the %s model..." %self.model)
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model == "poincare":
            model = MuRP(d, self.dim)
        else:
            model = MuRE(d, self.dim)
        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        
        if self.cuda:
            model.cuda()
            
        er_vocab = self.get_er_vocab(train_data_idxs)

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()

            losses = []
            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j:j+self.batch_size])
                negsamples = np.random.choice(list(self.entity_idxs.values()), 
                                              size=(data_batch.shape[0], self.nneg))
                
                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1]+1)))
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1]+1)))
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1))

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)      
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(it)
            print(time.time()-start_train)    
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it%5:
                    print("Test:")
                    self.evaluate(model, d.test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=40, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir)
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size, 
                            num_iterations=args.num_iterations, dim=args.dim, 
                            cuda=args.cuda, nneg=args.nneg, model=args.model)
    experiment.train_and_eval()
                

