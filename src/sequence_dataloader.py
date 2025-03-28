import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

class SequencePredictionDataset(Dataset):
    """
    A dataset that creates sliding windows for sequence prediction
    """
    def __init__(self, demos, window_size=20, prediction_offset=1):
        self.windows = []
        
        for demo in demos:
            images = [obs.wrist_rgb for obs in demo._observations]
            gripper_poses = [obs.gripper_pose for obs in demo._observations]
            

            images_tensor = torch.FloatTensor(np.array(images) / 255.0).permute(0, 3, 1, 2)  # Normalize and change to NCHW
            gripper_poses_tensor = torch.FloatTensor(np.array(gripper_poses))
            
            # Create sliding windows
            for start in range(0, len(images_tensor) - window_size - prediction_offset + 1):

                
                input_images = images_tensor[start:start+window_size] #0:20 #19
                input_poses = gripper_poses_tensor[start:start+window_size]
                
                target_pose = gripper_poses_tensor[start+window_size+prediction_offset-1]

                self.windows.append((
                    input_images,  
                    input_poses,   
                    target_pose     
                ))
    
    def __len__(self):
        """
        Returns the number of windows
        """
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Retrieve a specific window

        """
        return self.windows[idx]

def custom_sequence_prediction_collate_fn(batch):
    """
    Custom collate function to handle sequence prediction batches
    """
   
    images = [item[0] for item in batch]
    poses = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    
    batched_images = torch.stack(images)
    batched_poses = torch.stack(poses)
    batched_targets = torch.stack(targets)
    
    return batched_images, batched_poses, batched_targets

def create_sequence_prediction_dataloaders(
    demos, 
    window_size=20, 
    prediction_offset=1,
    batch_size=32, 
    num_workers=0, 
    train_ratio=0.8, 
    val_ratio=0.15
):
    """
    Create dataloaders for sequence prediction
    """

    sequence_dataset = SequencePredictionDataset(
        demos, 
        window_size=window_size, 
        prediction_offset=prediction_offset
    )

    total_size = len(sequence_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        sequence_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=custom_sequence_prediction_collate_fn
    )
    
    return train_loader, val_loader, test_loader
