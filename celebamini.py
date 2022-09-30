from torchvision.datasets.vision import VisionDataset
import PIL
import os
import torch
import pandas
import numpy as np

class CelebAMini(VisionDataset):
    """
    CelebA-mini is a subsample of the CelebA dataset 
    (<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)
    
    Args: 
        root (string): Root directory where the celeba-mini directory is
        transform (callable, optional): Input image transforms
        target_transform (callable, optional): Transform functions for the targets
        mode (string, optional): Mode to define the subset to load
    Dataset:
        2-tuple with:
         - sample: The input image (PIL.Image)
         - target: 2-tuple with:
           - gender: 0 for female, 1 for male
           - landmarks: 10x1 Tensor with:
             [0]: left eye x coordinate
             [1]: left eye y coordinate
             [2]: right eye x coordinate
             [3]: right eye y coordinate
             [4]: nose x coordinate
             [5]: nose y coordinate
             [6]: left mouth x coordinate
             [7]: left mouth y coordinate
             [8]: right mouth x coordinate
             [9]: right mouth y coordinate
    """

    base_folder = 'celeba-mini'
    img_folder = 'images'
    
    def __init__(self, root,
                 transform = None,
                 target_transform = None,
                 mode='train') :
        
        super(CelebAMini,self).__init__(root, transform=transform, 
                                        target_transform=target_transform)       

        self.mode = mode

        # check the mode is correct
        if mode not in ['train', 'val', 'test', 'all-train']:
            raise ValueError("Mode must be either 'train', 'val', 'test', or 'all-train'.")
        else:
            mode_name = '-' + mode # to append to file name

        # csv accordingly to mode
        self.csv_filename = f"celeba-mini{mode_name}.csv"

        #check the paths
        csv_path = os.path.join(self.root, self.base_folder, self.csv_filename)
        self.img_path = os.path.join(self.root, self.base_folder, self.img_folder)
        if not (os.path.isfile(csv_path) and os.path.isdir(self.img_path)):
            raise RuntimeError(f"Dataset not found in '{self.root}'.")
        

        #load the csv data
        csv_data = pandas.read_csv(csv_path, header=0, engine='python', index_col=0)
        
        self.filename = csv_data.index.values
        self.gender = torch.as_tensor(csv_data.iloc[:,0].values)
        self.landmarks = torch.as_tensor(csv_data.iloc[:, 1:].values)
        
        #verify images exist
        for f in self.filename:
            this_path = os.path.join(self.img_path, f)
            if not os.path.isfile(this_path):
                raise RuntimeError(f"Dataset corruption: Image '{this_path}' not found.")
    
    def __len__(self) :
        return len(self.filename)
    
    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.img_path, self.filename[index]))
        


        if self.transform is not None:

            input_transform = {'image': np.array(X), 'landmarks': self.landmarks[index,:].numpy().reshape(-1, 2)}
            output_transform = self.transform(input_transform)
            X = output_transform['image']
            T = (self.gender[index], output_transform['landmarks'].reshape(-1) / X.size[0]) 

        else:
            T = (self.gender[index], self.landmarks[index,:])

        return X, T                         