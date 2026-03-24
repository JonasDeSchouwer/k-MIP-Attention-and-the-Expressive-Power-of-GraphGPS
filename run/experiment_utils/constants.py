from dataclasses import dataclass
from enum import Enum
from typing import List

class ExecMode(Enum):
    PRINT = "print"
    TEST = "test"
    REAL = "real"

class Dataset(Enum):
    COCO = "COCO-SP"
    PeptStruct = "Peptides-struct"
    PeptFunc = "Peptides-func"
    Pascal = "PascalVOC"
    PCQM = "PCQM"
    Paris = "Paris"
    Shanghai = "Shanghai"
    LA = "LA"
    London = "London"
    ShapeNet = "ShapeNet"
    S3DIS = "S3DIS"

    def __str__(self):
        return self.value

    @classmethod
    def keys(cls) -> List[str]:
        return cls.__members__.keys()

    @classmethod
    def valid_identifiers(cls) -> List[str]:
        return ["all", "CityNetworks", "LRGB", "PointClouds"]

    @classmethod
    def identifier_to_dataset_list(cls, identifier: str) -> List["Dataset"]:
        if identifier == "all":
            return list(cls)
        elif identifier == "CityNetworks":
            return [d for d in cls if d.benchmark == "CityNetworks"]
        elif identifier == "LRGB":
            return [d for d in cls if d.benchmark == "LRGB"]
        elif identifier == "PointClouds":
            return [d for d in cls if d.benchmark == "PointClouds"]
        else:
            raise ValueError(f"Invalid dataset identifier: {identifier}")
        
    @property
    def benchmark(self) -> str:
        """Signifies the benchmark that the dataset belongs to.
        For example, "CityNetworks" or "LRGB".
        """
        if self in [Dataset.Paris, Dataset.Shanghai, Dataset.LA, Dataset.London]:
            return "CityNetworks"
        elif self in [Dataset.COCO, Dataset.Pascal, Dataset.PeptStruct, Dataset.PeptFunc]:
            return "LRGB"
        elif self == Dataset.PCQM:
            return None # we don't consider PCQM in our experiments
        elif self in [Dataset.ShapeNet, Dataset.S3DIS]:
            return "PointClouds"
        else:
            raise ValueError(f"Invalid dataset: {self}")

class Method(Enum):
    GAT = "GAT"
    GatedGCN = "GatedGCN"
    GCN = "GCN"
    GINE = "GINE"
    BigBird = "GPS+BigBird"
    kmip = "GPS+kmip"
    Performer = "GPS+Performer"
    Transformer = "GPS+Transformer"
    Exphormer = "Exphormer"

    def __str__(self):
        return self.value

    @classmethod
    def keys(cls) -> List[str]:
        return cls.__members__.keys()

    @classmethod
    def valid_identifiers(cls) -> List[str]:
        return ["all", "LRGB", "GTs", "GNNs"]

    @classmethod
    def identifier_to_method_list(cls, identifier: str) -> List["Method"]:
        if identifier == "all":
            return list(cls)
        elif identifier == "LRGB":
            return [cls.GAT, cls.BigBird, cls.kmip, cls.Performer]
        elif identifier == "GTs":
            return [cls.BigBird, cls.kmip, cls.Performer, cls.Transformer, cls.Exphormer]
        elif identifier == "GNNs":
            return [cls.GAT, cls.GatedGCN, cls.GCN, cls.GINE]
        else:
            raise ValueError(f"Invalid method identifier: {identifier}")
