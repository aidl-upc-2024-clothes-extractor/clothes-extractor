import os.path

from torch.utils.data import Dataset
from PIL import Image

# - image
# - agnostic-mask
# - cloth
# - cloth-mask

class ClothesDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        return
