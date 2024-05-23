from .mf_dataset import MFDataset

class DCNDataset(MFDataset):
    def __init__(self, data, num_items=None): 
        super().__init__(data, num_items)
