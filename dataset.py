import os.path as osp
import torch
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import degree
from torch_geometric.io import read_tu_data
import torch
from typing import Optional, Callable, List
import numpy as np
import os
import shutil
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data


class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
            
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            
        if self.name == "REDDIT-BINARY" or self.name == "COLLAB" or self.name == "IMDB-BINARY" or self.name == "IMDB-MULTI" or self.name== "Synthie" or self.name== "FRANKENSTEIN":
            self.data, self.slices = torch.load(self.processed_paths[0])
            node_features = []
            x_slices = [0]
            for adj_t in self.data.adj_t:
                row, col, _ = adj_t.coo() 
                edge_index = torch.stack([row, col], dim=0)  
                num_nodes = adj_t.size(0)  
                deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
                x = torch.ones((num_nodes, 1)) 
                node_features.append(x)
                x_slices.append(x_slices[-1] + num_nodes)
            self.data.x = torch.cat(node_features, dim=0)
            self.slices['x'] = torch.tensor(x_slices, dtype=torch.long)
        
        self._load_or_compute_inductive_pre()
   

    def _load_or_compute_inductive_pre(self):
        """Add node-level structural info: [graph_size, avg_deg, avg_edges, node_deg]"""
        inductive_pre_path = os.path.join(self.processed_dir, 'inductive_pre.pt')
        slices_path = os.path.join(self.processed_dir, 'slices.pt')

        if os.path.exists(inductive_pre_path) and os.path.exists(slices_path):
            print(f"Loading precomputed inductive_pre for {self.name}...")
            self.data.inductive_pre = torch.load(inductive_pre_path)
            self.slices = torch.load(slices_path)
        else:
            print(f"Computing inductive_pre for {self.name}...")
            node_indices = []
            inductive_pre_slices = [0]

            for i in range(len(self)):
                data = self.get(i)
                row, col, _ = data.adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                num_nodes = data.num_nodes
                deg = degree(edge_index[0], num_nodes=num_nodes)

                num_edges = edge_index.size(1) // 2  
                avg_deg = deg.float().mean().item()
                avg_edges = num_edges / num_nodes

            
                graph_inductive_pre = torch.stack([
                    torch.full((num_nodes,), num_nodes),         
                    torch.full((num_nodes,), avg_deg),         
                    torch.full((num_nodes,), avg_edges),         
                    deg.float()                                   
                ], dim=1) 

                node_indices.append(graph_inductive_pre)
                inductive_pre_slices.append(inductive_pre_slices[-1] + num_nodes)

            self.data.inductive_pre = torch.cat(node_indices, dim=0)  
            self.slices['inductive_pre'] = torch.tensor(inductive_pre_slices, dtype=torch.long)

            torch.save(self.data.inductive_pre, inductive_pre_path)
            torch.save(self.slices, slices_path)
            print(f"inductive_pre saved to {inductive_pre_path}")

        print(self.data)
        print(torch.unique(self.data.y))
        self.data.id = torch.arange(0, self.data.y.size(0))
        self.slices['id'] = self.slices['y'].clone()
 

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)  

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)  
    
    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels


    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices,_ = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def get_TUDataset(dataset, pre_transform):
    """
    'PROTEINS', 'REDDIT-BINARY', 'MUTAG', 'PTC_MR', 'DD', 'NCI1'
    """
    path = osp.join("/U2B_AAAI", 'data', 'TU')  
    dataset = TUDataset(path, name=dataset, pre_transform=pre_transform)
    n_feat, n_class = max(dataset.num_features, 1), dataset.num_classes
    mapping = {}
    return dataset, n_feat, n_class, mapping


def shuffle(dataset, c_train_num, c_val_num, y):
    classes = torch.unique(y)
    indices = []
    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)  
        index = index[torch.randperm(index.size(0))]
        indices.append(index)  
    train_index, val_index, test_index = [], [], []
    for i in range(len(classes)):
        train_index.append(indices[classes[i]][:c_train_num[classes[i]]])
        val_index.append(indices[classes[i]][c_train_num[classes[i]]:(c_train_num[classes[i]] + c_val_num[classes[i]])])
        test_index.append(indices[classes[i]][(c_train_num[classes[i]] + c_val_num[classes[i]]):])
    train_index = torch.cat(train_index, dim=0)
    val_index = torch.cat(val_index, dim=0)
    test_index = torch.cat(test_index, dim=0)
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    return train_dataset, val_dataset, test_dataset

def load_split(load_path='', split_mode='low', load_file=None):
    if load_file:
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Parameter load_file is not a valid file")
    elif os.path.exists(load_path):
        load_file = os.path.join(load_path, 'split_' + split_mode + '.pt')
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Cannot find split.pt in load_path")
    else:
        raise ValueError("Fail to load split file, please check parameter load_path or load_file")
    train_mask = loaded_split['train_mask']
    val_mask = loaded_split['val_mask']
    test_mask = loaded_split['test_mask']
    boundary_size = loaded_split['boundary_size']
    return train_mask, val_mask, test_mask, boundary_size


def cal_imbalance_ratio(dataset, boundary_size):
    head_size = []
    tail_size = []
    for g in dataset:
        if g.num_nodes <= boundary_size:
            tail_size.append(g.num_nodes)
        else:
            head_size.append(g.num_nodes)

    head_avg = round(np.mean(head_size), 4)
    tail_avg = round(np.mean(tail_size), 4)
    imbalance_ratio = round(head_avg / tail_avg, 4)
    return head_avg, tail_avg, imbalance_ratio