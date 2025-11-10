import math
import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score
import os
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from dataprocess import *
from dataset import *
from sklearn.metrics import f1_score
import time
from colorama import Fore, init
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from U2B.U2B import U2B
import random
init(autoreset=True)
def runnerr(args, device):
    
    log_dir = f'/U2B_AAAI/log/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")  
    log_filename = f'{log_dir}/{args.dataset}_training_log_{timestamp}.txt'
    with open(log_filename, 'w') as log_file:
        log_file.write("Training hyperparameters:\n")
        for key, value in vars(args).items():
            log_file.write(f'{key}: {value}\n')
        log_file.write("\nTraining begins...\n")

    F1_micro = np.zeros(args.runs, dtype=float)
    F1_macro = np.zeros(args.runs, dtype=float)
    AUROC = np.zeros(args.runs, dtype=float)
    Accuracy = np.zeros(args.runs, dtype=float)
    Balanced_accuracy = np.zeros(args.runs, dtype=float)
    tail_acc = np.zeros(args.runs, dtype=float)
    tail_f1 = np.zeros(args.runs, dtype=float)
    head_acc = np.zeros(args.runs, dtype=float)
    head_f1 = np.zeros(args.runs, dtype=float)
    
    for count in range(args.runs):
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        train_mask, val_mask, test_mask, boundary_size = load_split(osp.join("/U2B_AAAI/data/TU", args.dataset), split_mode= args.split_mode)
        print("boundary_size:",boundary_size)
        train_dataset = dataset[train_mask]
        val_dataset = dataset[val_mask]
        test_dataset = dataset[test_mask]
        head_avg, tail_avg, imbalance_ratio = cal_imbalance_ratio(train_dataset, boundary_size)
        print("head_avg:", head_avg)
        print("tail_avg:", tail_avg)
        print("imbalance_ratio:", imbalance_ratio)
        print(len(train_dataset))
        print(len(val_dataset))
        train_dataset = Dataset(train_dataset, dataset, args)
        val_dataset = Dataset(val_dataset, dataset, args)
        test_dataset = Dataset(test_dataset, dataset, args)
        if args.dataset == 'FRANKENSTEIN' :
            shuffle_list = [False, False, True]
        else:
            shuffle_list = [True, False, False]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_list[0], collate_fn=train_dataset.collate_batch)  
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle_list[1], collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle_list[2], collate_fn=test_dataset.collate_batch)
        model = U2B(args, args.n_hidden, args.num_gc_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
        best_val_loss = math.inf  
        val_loss_hist = []
        bias = torch.zeros(args.K_g).to(device)
        for epoch in tqdm(range(0, args.epochs)):
            loss, topk = train(model, train_loader, optimizer, args, device, epoch + 1, scheduler, bias)
            val_eval = eval(model, val_loader, args, device, bias)
                
            if val_eval['loss'] < best_val_loss:
                best_val_loss = val_eval['loss']
                test_eval = eval(model, test_loader, args, device, bias)
                
            val_loss_hist.append(val_eval['loss'])
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = torch.tensor(val_loss_hist[-(args.early_stopping + 1): -1])
                if val_eval['loss'] > tmp.mean().item():
                    break
            mu = args.topk * args.batch_size / args.K_g
            x = topk - mu
            delta = 0.05 * mu
            abs_x = torch.abs(x)
            grad = torch.where(abs_x <= delta, x, delta * torch.sign(x))
            bias = bias - 0.001 * grad
        F1_macro[count] = test_eval['F1-macro']
        AUROC[count] = test_eval['AUROC']
        Balanced_accuracy[count] = test_eval['Balanced Accuracy']
        Accuracy[count] = test_eval['Accuracy']
        tail_acc[count] = test_eval['Tail Accuracy']
        tail_f1[count] = test_eval["Tail F1"]
        head_acc[count] = test_eval['Head Accuracy']
        head_f1[count] = test_eval["Head F1"]
        
        print(Fore.CYAN + f"Run {count + 1}: F1_macro = {F1_macro[count]}, AUROC = {AUROC[count]}, Balanced_accuracy= {Balanced_accuracy[count]}, Accuracy = {Accuracy[count]}, tail_acc = {tail_acc[count]}, tail_f1 = {tail_f1[count]}, head_acc = {head_acc[count]}, head_f1 = {head_f1[count]}")
        print(Fore.YELLOW + f"Best validation loss: {best_val_loss}")
        print(Fore.MAGENTA + f"Average F1_macro: {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})")
        print(Fore.MAGENTA + f"Average AUROC: {np.mean(AUROC[:count+1])} (std = {np.std(AUROC[:count+1])})")
        print(Fore.MAGENTA + f"Average Balanced_accuracy: {np.mean(Balanced_accuracy[:count+1])} (std = {np.std(Balanced_accuracy[:count+1])})")
        print(Fore.MAGENTA + f"Average Accuracy: {np.mean(Accuracy[:count+1])} (std = {np.std(Accuracy[:count+1])})")
        print(Fore.MAGENTA + f"Average tail_acc: {np.mean(tail_acc[:count+1])} (std = {np.std(tail_acc[:count+1])})")
        print(Fore.MAGENTA + f"Average tail_f1: {np.mean(tail_f1[:count+1])} (std = {np.std(tail_f1[:count+1])})")
        print(Fore.MAGENTA + f"Average head_acc: {np.mean(head_acc[:count+1])} (std = {np.std(head_acc[:count+1])})")
        print(Fore.MAGENTA + f"Average head_f1: {np.mean(head_f1[:count+1])} (std = {np.std(head_f1[:count+1])})")
        
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\nRun {count+1}: F1_macro = {F1_macro[count]}, AUROC = {AUROC[count]}, Balanced_accuracy= {Balanced_accuracy[count]}, Accuracy = {Accuracy[count]}, tail_acc = {tail_acc[count]}, tail_f1 = {tail_f1[count]}, head_acc = {head_acc[count]}, head_f1 = {head_f1[count]}\n")
            log_file.write(f"Best validation loss: {best_val_loss}\n")
            log_file.write(f"Average F1_macro = {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})\n")
            log_file.write(f"Average AUROC: {np.mean(AUROC[:count+1])} (std = {np.std(AUROC[:count+1])})\n")
            log_file.write(f"Average Balanced_accuracy: {np.mean(Balanced_accuracy[:count+1])} (std = {np.std(Balanced_accuracy[:count+1])})\n")
            log_file.write(f"Average Accuracy: {np.mean(Accuracy[:count+1])} (std = {np.std(Accuracy[:count+1])})\n")
            log_file.write(f"Average tail_acc: {np.mean(tail_acc[:count+1])} (std = {np.std(tail_acc[:count+1])})\n")
            log_file.write(f"Average tail_f1: {np.mean(tail_f1[:count+1])} (std = {np.std(tail_f1[:count+1])})\n")
            log_file.write(f"Average head_acc: {np.mean(head_acc[:count+1])} (std = {np.std(head_acc[:count+1])})\n")
            log_file.write(f"Average head_f1: {np.mean(head_f1[:count+1])} (std = {np.std(head_f1[:count+1])})\n")
            
    return  F1_macro, AUROC, Balanced_accuracy, Accuracy, tail_acc, tail_f1, head_acc, head_f1

def train(model, data_loader, optimizer, args, device, epoch, scheduler, bias):
    model.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):
        batch_to_gpu(batch, device)
        data, train_idx = batch['data'], batch['train_idx']
        batch_size = batch['data'].y.shape[0]
        if data.edge_index is None and data.adj_t is not None:
                data.edge_index = torch.stack(data.adj_t.coo()[:2], dim=0)
        losss, logits, topk_indices = model(data.x, data.adj_t, data.batch, data.inductive_pre, bias)
        logits = logits[train_idx]
        loss = F.nll_loss(logits, data.y[train_idx]) + losss
        total_loss = loss * batch_size
        topk = torch.bincount(topk_indices.flatten(), minlength=args.K_g).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler:
            scheduler.step()
        avg_train_loss = total_loss / (i + 1)
    return avg_train_loss, topk

def eval(model, data_loader, args, device, bias):
    model.eval()
    pred, truth = [], []
    probas = []
    total_loss = 0
    tail_pred, tail_truth = [], []
    head_pred, head_truth = [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch_to_gpu(batch, device)
            data, train_idx = batch['data'], batch['train_idx']
            
            if data.edge_index is None and data.adj_t is not None:
                data.edge_index = torch.stack(data.adj_t.coo()[:2], dim=0)

            loss_val, logits, _ = model(data.x, data.edge_index, data.batch, data.inductive_pre, bias)
            loss = F.nll_loss(logits, data.y)
            total_loss += (loss * train_idx.shape[0]).item()
            probs = torch.exp(logits)
            pred_batch = logits.argmax(dim=-1)
            truth_batch = data.y
            pred.extend(pred_batch.tolist())
            truth.extend(truth_batch.tolist())
            probas.extend(probs.tolist())
            num_nodes_per_graph = torch.bincount(data.batch)
            for j in range(len(num_nodes_per_graph)):
                if num_nodes_per_graph[j].item() < args.K:
                    tail_pred.append(pred_batch[j].item())
                    tail_truth.append(truth_batch[j].item())
                else:
                    head_pred.append(pred_batch[j].item())
                    head_truth.append(truth_batch[j].item())
    pred_np = np.array(pred)
    truth_np = np.array(truth)
    probas_np = np.array(probas)

    acc = accuracy_score(truth_np, pred_np)
    f1_macro = f1_score(truth_np, pred_np, average='macro', zero_division=0)
    f1_micro = f1_score(truth_np, pred_np, average='micro', zero_division=0)
    balanced_acc = balanced_accuracy_score(truth_np, pred_np)

    try:
        auroc = roc_auc_score(
            np.eye(args.n_class)[truth_np],
            probas_np,
            multi_class='ovr'
        )
    except ValueError:
        auroc = -1
    if len(tail_truth) > 0:
        tail_acc = accuracy_score(tail_truth, tail_pred)
        tail_f1 = f1_score(tail_truth, tail_pred, average='macro', zero_division=0)
    else:
        tail_acc = -1
        tail_f1 = -1
    if len(head_truth) > 0:
        head_acc = accuracy_score(head_truth, head_pred)
        head_f1 = f1_score(head_truth, head_pred, average='macro', zero_division=0)
    else:
        head_acc = -1
        head_f1 = -1

    return {
        'loss': total_loss / (i + 1),
        'F1-macro': f1_macro,
        'F1-micro': f1_micro,
        'Accuracy': acc,
        'Balanced Accuracy': balanced_acc,
        'AUROC': auroc,
        'Tail Accuracy': tail_acc,
        'Tail F1': tail_f1,
        'Head Accuracy': head_acc,
        'Head F1': head_f1,
    }

if __name__ == '__main__':
    args = parse_args()
    args.num_gc_layers = 5
    if args.dataset == 'PTC':
        args.K = 22
    elif args.dataset == 'PTC_MR':
        args.K = 19
    elif args.dataset == 'PROTEINS':
        args.K = 54
    elif args.dataset == 'IMDB-BINARY':
        args.K = 25
    elif args.dataset == 'DD':
        args.K = 395
    elif args.dataset == 'FRANKENSTEIN':
        args.K = 22
    elif args.dataset == 'REDDIT-BINARY':
        args.K = 469
    elif args.dataset == 'COLLAB':
        args.K = 91
    elif args.dataset == 'NCI1':
        args.K = 38
    elif args.dataset == 'IMDB-MULTI':
        args.K = 17

    args.path = os.getcwd()
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    path = osp.join("/U2B_AAAI", 'data', 'TU')  
    dataset, args.n_feat, args.n_class, _ = get_TUDataset(args.dataset, pre_transform=T.ToSparseTensor())
    args.dataset_num_features = max(dataset.num_features, 1)
    F1_macro, AUROC, Balanced_accuracy, Accuracy, tail_acc, tail_f1, head_acc, head_f1 = runnerr(args, device)  
    print('F1_macro: ', np.mean(F1_macro), np.std(F1_macro))
    print('AUROC: ', np.mean(AUROC), np.std(AUROC))
    print('Balanced_accuracy: ', np.mean(Balanced_accuracy), np.std(Balanced_accuracy))
    print('Accuracy: ', np.mean(Accuracy), np.std(Accuracy))
    print('tail_acc: ', np.mean(tail_acc), np.std(tail_acc))
    print('tail_f1: ', np.mean(tail_f1), np.std(tail_f1))
    print('head_acc: ', np.mean(head_acc), np.std(head_acc))
    print('head_f1: ', np.mean(head_f1), np.std(head_f1))