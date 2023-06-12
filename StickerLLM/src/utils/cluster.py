import os
# from pyotrch_lightning.pytorch.plugins.environments import ClusterEnvironment
from pytorch_lightning.plugins.environments import ClusterEnvironment


class MyClusterEnvironment(ClusterEnvironment):

    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    def main_address(self) -> str:
        return os.environ["MASTER_ADDRESS"]

    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])
