from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the CityNetworks dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Node attributes: 12
    - longtitude
    - latitude
    - street count
    - land use (9 categories)
Edge attributes: 25 (added to adjacent nodes)
    - road length in metres
    - speed limit in km/h
    - one-way or not (binary)
    - reversed or not (binary)
    - lane count (9 categories)
    - road type (9 categories)
Shape of node features: [num_nodes, 37]
Edge features: None
Shape of y : [num_nodes] (elements: 0 to 9)
"""


# obtained through:
# t = batch.x[batch.train_mask]
# node_x_mean = t.mean(dim=0)
# node_x_std = t.std(dim=0)

@dataclass
class DatasetValues:
    node_x_mean: torch.Tensor
    node_x_std: torch.Tensor
    train_longitude_range: tuple[float, float]
    train_latitude_range: tuple[float, float]

Paris_values = DatasetValues(
    node_x_mean=torch.tensor([
        2.3425e+00, 4.8858e+01, 3.2222e+00, 9.5023e-01, 1.7000e-02, 1.4283e-02,
        7.4483e-03, 6.3091e-03, 1.6649e-03, 1.6649e-03, 3.5051e-04, 1.0515e-03,
        3.1304e+01, 2.5179e+01, 7.8846e-01, 2.1154e-01, 9.1035e-01, 3.8676e-02,
        2.8768e-02, 1.0248e-02, 8.1668e-03, 1.2516e-03, 1.0063e-03, 1.1318e-03,
        4.0162e-04, 5.5726e-01, 3.4572e-01, 9.7024e-02, 5.9242e-01, 1.1367e-01,
        7.9949e-02, 7.0175e-02, 3.8595e-02, 2.9458e-02, 2.8289e-02, 2.5261e-02,
        2.2192e-02
    ]),
    node_x_std=torch.tensor([
        4.2885e-02, 2.0501e-02, 7.9370e-01, 2.1748e-01, 1.2928e-01, 1.1866e-01,
        8.5985e-02, 7.9183e-02, 4.0771e-02, 4.0771e-02, 1.8719e-02, 3.2412e-02,
        2.8403e+01, 3.5520e+00, 3.2239e-01, 3.2239e-01, 2.2383e-01, 1.4168e-01,
        1.2640e-01, 7.3338e-02, 6.6672e-02, 2.5828e-02, 2.2214e-02, 1.9059e-02,
        1.2003e-02, 1.8950e-01, 1.7185e-01, 1.6418e-01, 4.0623e-01, 2.4347e-01,
        2.3066e-01, 2.1130e-01, 1.4823e-01, 1.3038e-01, 1.3556e-01, 1.1891e-01,
        1.1273e-01
    ]),
    train_longitude_range=(2.226167917251587, 2.4697108268737793),
    train_latitude_range=(48.81593322753906, 48.90190124511719)
)

Shanghai_values = DatasetValues(
    node_x_mean=torch.tensor([
        1.2143e+02, 3.1165e+01, 2.8890e+00, 4.1134e-01, 1.2473e-01, 1.2365e-01,
        8.4878e-02, 7.3949e-02, 5.7637e-02, 5.4809e-02, 4.2575e-02, 2.6426e-02,
        1.6883e+02, 3.6067e+01, 8.2906e-01, 1.7094e-01, 9.3413e-01, 3.1887e-02,
        9.7418e-03, 1.3953e-02, 6.8394e-03, 1.3897e-03, 1.3086e-03, 2.6190e-04,
        4.8846e-04, 5.8091e-01, 4.0997e-01, 9.1177e-03, 2.6486e-01, 2.3413e-01,
        1.0017e-01, 8.6092e-02, 7.8679e-02, 8.9317e-02, 5.9594e-02, 4.8329e-02,
        3.8830e-02
    ]),
    node_x_std=torch.tensor([
        2.3066e-01, 1.8717e-01, 9.7593e-01, 4.9209e-01, 3.3043e-01, 3.2919e-01,
        2.7871e-01, 2.6170e-01, 2.3306e-01, 2.2761e-01, 2.0190e-01, 1.6040e-01,
        2.2267e+02, 9.0400e+00, 3.2411e-01, 3.2411e-01, 2.0172e-01, 1.3452e-01,
        7.6818e-02, 8.8936e-02, 6.2548e-02, 2.7983e-02, 2.2116e-02, 1.1592e-02,
        1.2711e-02, 1.6681e-01, 1.6315e-01, 6.2032e-02, 4.0305e-01, 3.7558e-01,
        2.6771e-01, 2.2543e-01, 2.3196e-01, 2.4780e-01, 1.8691e-01, 1.7188e-01,
        1.8102e-01
    ]),
    train_longitude_range=(120.85847473144531, 121.96424102783203),
    train_latitude_range=(30.700109481811523, 31.857986450195312)
)

LA_values = DatasetValues(
    node_x_mean=torch.tensor([
        -1.1840e+02,  3.4122e+01,  2.8899e+00,  5.1417e-01,  1.3709e-01,
        9.9842e-02,  7.7230e-02,  5.7735e-02,  5.5865e-02,  2.8307e-02,
        1.9952e-02,  9.8096e-03,  6.5346e+01,  3.3122e+01,  9.2056e-01,
        7.9437e-02,  8.3675e-01,  7.7101e-02,  2.2165e-02,  1.4550e-02,
        1.6315e-02,  1.2191e-02,  6.6111e-03,  5.6696e-03,  8.6444e-03,
        5.3363e-01,  4.5419e-01,  1.2184e-02,  3.7541e-01,  2.1603e-01,
        2.0149e-01,  4.8105e-02,  4.5459e-02,  4.6159e-02,  4.2846e-02,
        1.3700e-02,  1.0802e-02
    ]),
    node_x_std=torch.tensor([
        1.2046e-01, 1.0232e-01, 9.9945e-01, 4.9981e-01, 3.4394e-01, 2.9980e-01,
        2.6696e-01, 2.3325e-01, 2.2967e-01, 1.6585e-01, 1.3984e-01, 9.8559e-02,
        7.2480e+01, 1.1345e+01, 2.1460e-01, 2.1460e-01, 3.0637e-01, 2.2009e-01,
        1.0960e-01, 8.7333e-02, 8.5987e-02, 7.8526e-02, 5.1131e-02, 5.3880e-02,
        7.3521e-02, 1.1494e-01, 1.1218e-01, 7.4421e-02, 4.0856e-01, 3.4994e-01,
        3.4596e-01, 1.6835e-01, 1.6329e-01, 1.6530e-01, 1.6942e-01, 1.0543e-01,
        8.3749e-02
    ]),
    train_longitude_range=(-118.66815185546875, -118.15784454345703),
    train_latitude_range=(33.84650421142578, 34.33503341674805)
)

London_values = DatasetValues(
    node_x_mean=torch.tensor([
        -1.1294e-01,  5.1498e+01,  2.7004e+00,  6.6309e-01,  8.7238e-02,
        7.0026e-02,  5.9952e-02,  4.0947e-02,  3.7272e-02,  1.4522e-02,
        1.3256e-02,  1.3696e-02,  4.4827e+01,  3.3014e+01,  9.1421e-01,
        8.5793e-02,  9.2041e-01,  5.6466e-02,  1.1682e-02,  6.5663e-03,
        2.2724e-03,  9.4587e-04,  6.9724e-04,  6.9944e-04,  2.5991e-04,
        5.3162e-01,  4.4582e-01,  2.2563e-02,  2.5665e-01,  2.3231e-01,
        2.7018e-01,  7.9964e-02,  3.6583e-02,  3.8723e-02,  3.2653e-02,
        2.6432e-02,  2.6497e-02
    ]),
    node_x_std=torch.tensor([
        1.6439e-01, 7.8307e-02, 9.8483e-01, 4.7266e-01, 2.8219e-01, 2.5519e-01,
        2.3740e-01, 1.9817e-01, 1.8943e-01, 1.1963e-01, 1.1437e-01, 1.1623e-01,
        4.6011e+01, 4.8145e+00, 2.3295e-01, 2.3295e-01, 2.2161e-01, 1.8168e-01,
        8.2169e-02, 5.9227e-02, 3.4996e-02, 1.7517e-02, 1.6765e-02, 1.6332e-02,
        1.1128e-02, 1.2910e-01, 1.2450e-01, 1.0034e-01, 3.8191e-01, 3.6175e-01,
        3.9332e-01, 2.1473e-01, 1.5228e-01, 1.5814e-01, 1.4354e-01, 1.4488e-01,
        1.3209e-01
    ]),
    train_longitude_range=(-0.5015630125999451, 0.3162260949611664),
    train_latitude_range=(51.28840255737305, 51.689353942871094)
)

DATASET_VALUES = {
    "Paris": Paris_values,
    "Shanghai": Shanghai_values,
    "LA": LA_values,
    "London": London_values
}


@register_node_encoder('CityNetworkNode')
class CityNetworkNodeEncoder(torch.nn.Module):
    node_input_dim = 37
    num_freqs = 8
    safety_factor = 0.9

    def __init__(self, emb_dim):
        super().__init__()
        
        dataset_name = cfg.dataset.name
        self.values = DATASET_VALUES[dataset_name]

        self.register_buffer('node_x_mean', self.values.node_x_mean)
        self.register_buffer('node_x_std', self.values.node_x_std)
        self.encoder = torch.nn.Linear(self.node_input_dim - 2 + 4*self.num_freqs, emb_dim)

    def forward(self, batch):
        x = self.encode_lat_lon(batch.x)
        x[:, 4*self.num_freqs:] = self.normalize(x[:, 4*self.num_freqs:], self.node_x_mean[2:], self.node_x_std[2:])
        batch.x = self.encoder(x)
        return batch

    def normalize(self, x, mean_, std_):
        """Args:
            x: torch.Tensor, shape (n, d)
            mean_: torch.Tensor, shape (d,)
            std_: torch.Tensor, shape (d,)

        Returns:
            torch.Tensor, shape (n, d)
        """
        x = (x - mean_) / std_
        return x

    def encode(self, v, min_value=None, max_value=None):
        """Encode nx1 vector v using sines and cosines
        
        Args:
            v: torch.Tensor, shape (n, 1)
            min_value: float, minimum value of v
            max_value: float, maximum value of v

        Returns:
            torch.Tensor, shape (n, 2*num_freqs)
        """

        # normalize x to be in [-pi, pi], based on values from t (to avoid data leakage)
        v = (v - min_value) / (max_value - min_value)   # [0, 1]
        v = v - 0.5                                     # [-0.5, 0.5]
        v *= 2 * np.pi * self.safety_factor             # [-pi * safety_factor, pi * safety_factor]

        # encode x using sines and cosines
        sines = torch.cat([torch.sin(v * (2**i)) for i in range(self.num_freqs)], dim=1)
        cosines = torch.cat([torch.cos(v * (2**i)) for i in range(self.num_freqs)], dim=1)

        return torch.cat([sines, cosines], dim=1)


    def encode_lat_lon(self, x):
        """Change x to have sine-cosine encoded longitude and latitude.
        
        Args:
            x: torch.Tensor, shape (n, d) containing node features where first two columns are longitude and latitude
            
        Returns:
            new_x: torch.Tensor, shape (n, d) containing node features where:
                first 2*num_freqs columns are sine-cosine encoded longitude
                the next 2*num_freqs columns are sine-cosine encoded latitude
                the rest of the columns are the original node features
        """
        x_lon = x[:, 0].unsqueeze(1)
        x_lat = x[:, 1].unsqueeze(1)

        lon_range = self.values.train_longitude_range
        lat_range = self.values.train_latitude_range
        x_lon_enc = self.encode(x_lon, min_value=lon_range[0], max_value=lon_range[1])
        x_lat_enc = self.encode(x_lat, min_value=lat_range[0], max_value=lat_range[1])

        new_x = torch.cat([x_lon_enc, x_lat_enc, x[:, 2:]], dim=1)

        return new_x
    
@register_edge_encoder('CityNetworkEdge')
class CityNetworkEdgeEncoder(torch.nn.Module):
    edge_input_dim = 25

    def __init__(self, emb_dim):
        super().__init__()

        raise Exception("Edge encoder not implemented for CityNetworks")

        self.encoder = torch.nn.Linear(self.edge_input_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch