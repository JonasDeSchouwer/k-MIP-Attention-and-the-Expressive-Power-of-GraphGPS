"""
Credits to https://github.com/LeonResearch/City-Networks/blob/main/citynetworks.py
"""


import io
import os.path as osp
import pickle
import re
import shutil
from typing import Any, Callable, Optional, Sequence, Type
import warnings

import torch

import torch_geometric
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar
)
from torch_geometric.data.data import BaseData

class CityNetwork(InMemoryDataset):
    r"""The City-Networks are introduced in
    `"Towards Quantifying Long-Range Interactions in Graph Machine Learning:
    a Large Graph Dataset and a Measurement"
    <https://arxiv.org/abs/2503.09008>`_ paper.
    The dataset contains four city networks: `paris`, `shanghai`, `la`,
    and 'london', where nodes represent junctions and edges represent
    directed road segments. The task is to predict each node's eccentricity
    score, which is approximated based on its 16-hop neighborhood. The score
    indicates how accessible one node is in the network, and is mapped to
    10 quantiles for transductive classification. See the original
    `source code <https://github.com/LeonResearch/City-Networks>`_ for more
    details on the individual networks.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"paris"`,
        :obj:`"shanghai"`, :obj:`"la"`, :obj:`"london"`).
        augmented (bool, optional): Whether to use the augmented node features
        from edge features.
        (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**
    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - paris
          - 114,127
          - 182,511
          - 37
          - 10
        * - shanghai
          - 183,917
          - 262,092
          - 37
          - 10
        * - la
          - 240,587
          - 341,523
          - 37
          - 10
        * - london
          - 568,795
          - 756,502
          - 37
          - 10
    """

    url = ("https://github.com/LeonResearch/"
           "City-Networks/raw/refs/heads/main/data/")

    def __init__(
        self,
        root: str,
        name: str,
        augmented: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        delete_raw: bool = False,
    ) -> None:
        self.name = name.lower()
        assert self.name in ["paris", "shanghai", "la", "london"]
        self.augmented = augmented
        self.delete_raw = delete_raw

        if force_reload:
            shutil.rmtree(self.processed_dir)

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.json"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"
    
    @classmethod
    def torch_save(cls, data: Any, path: str) -> None:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        with open(path, 'wb') as f:
            f.write(buffer.getvalue())
    
    @classmethod
    def save(cls, data_list: Sequence[BaseData], path: str) -> None:
        r"""Saves a list of data objects to the file path :obj:`path`."""
        data, slices = cls.collate(data_list)
        cls.torch_save((data.to_dict(), slices, data.__class__), path)

    @classmethod
    def torch_load(cls, path: str) -> Any:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=None)

    def load(self, path: str, data_cls: Type[BaseData] = Data) -> None:
        r"""Loads the dataset from the file path :obj:`path`."""
        out = self.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) == 2 or len(out) == 3
        if len(out) == 2:  # Backward compatibility.
            data, self.slices = out
        else:
            data, self.slices, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def download(self) -> None:
        self.download_path = download_url(self.url + f"{self.name}.tar.gz",
                                          self.raw_dir)

    def process(self) -> None:
        extract_tar(self.download_path, self.raw_dir)
        data_path = osp.join(self.raw_dir, self.name)
        node_feat = (torch.load(
            osp.join(data_path, "node_features_augmented.pt")) if self.augmented else torch.load(
                osp.join(data_path, "node_features.pt")))
        edge_index = torch.load(osp.join(data_path, "edge_indices.pt"))
        label = torch.load(
            osp.join(data_path, "10-chunk_16-hop_node_labels.pt"))

        train_mask = torch.load(osp.join(data_path, "train_mask.pt"))
        val_mask = torch.load(osp.join(data_path, "valid_mask.pt"))
        test_mask = torch.load(osp.join(data_path, "test_mask.pt"))

        # test if train, val, test are disjoint
        assert (train_mask & val_mask).sum() == 0
        assert (train_mask & test_mask).sum() == 0
        assert (val_mask & test_mask).sum() == 0
        assert train_mask.sum() + val_mask.sum() + test_mask.sum() == node_feat.shape[0]

        data = Data(
            x=node_feat,
            edge_index=edge_index,
            y=label,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        if self.delete_raw:
            shutil.rmtree(data_path)

    def get_idx_split(self):
        return {
            'train': torch.nonzero(self.data.train_mask, as_tuple=False).squeeze(),
            'val': torch.nonzero(self.data.val_mask, as_tuple=False).squeeze(),
            'test': torch.nonzero(self.data.test_mask, as_tuple=False).squeeze()
        }


if __name__ == '__main__':
    dataset = CityNetwork(root='datasets', name='la')
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[0])
    print(dataset[0].y)
    print(dataset.get_idx_split())
    for k,v in dataset.get_idx_split().items():
        print(k, v.shape, v.sum())