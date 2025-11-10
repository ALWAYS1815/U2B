import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="MUTAG",
                        help="Choose a dataset:[MUTAG, PROTEINS, DHFR, DD, NCI1, PTC-MR, REDDIT-B]")
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    # model
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--prop_epochs', type=int, default=3)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder_seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.005)

    parser.add_argument('--alpha', type=float, default = 1.0)
    parser.add_argument('--head', type=int, default = 4)
    parser.add_argument('--K_v', type=int, default = 8)
    parser.add_argument('--K_g', type=int, default = 8)
    parser.add_argument('--topk', type=int, default = 8)
    parser.add_argument('--dim', type=int, default = 128)
    parser.add_argument('--n_hidden', type=int, default=32)

    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--aug', type=str, default='dnodes')
    
    # Scheduler
    parser.add_argument('--step_size', type=int, default= 10)
    parser.add_argument('--gamma', type=int, default= 0.5)
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--memory_error', type=bool, default=False)

    # Split
    parser.add_argument('--split_mode', type=str, default="high")
    # setting
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--early_stopping', type=int, default=100)
    return parser.parse_args()
