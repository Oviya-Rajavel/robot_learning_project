import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import math
import os
import matplotlib.pyplot as plt
import io
import numpy as np
import matplotlib.image as mpimg

import torch
import numpy as np
import matplotlib.pyplot as plt
import io

def plot_trajectory(poses, predicted_poses, target_poses, prediction_horizon=1):
    """
    Create a 3D trajectory plot for TensorBoard visualization
    """

    poses = poses.cpu().numpy()
    predicted_poses = predicted_poses.cpu().numpy()
    target_poses = target_poses.cpu().numpy()
    

    if predicted_poses.ndim > 2:
        predicted_poses = predicted_poses[0]
    if target_poses.ndim > 2:
        target_poses = target_poses[0]
    

    trajectory = poses[:, :, :3]
    

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    

    ax.plot(trajectory[0, :, 0], trajectory[0, :, 1], trajectory[0, :, 2], 'b-o', label='Input Trajectory')
    


    last_point = trajectory[0, prediction_horizon-1]

    

    predicted_point = predicted_poses[:3] if predicted_poses.ndim == 1 else predicted_poses[0, :3]
    target_point = target_poses[:3] if target_poses.ndim == 1 else target_poses[0, :3]

    ax.plot([last_point[0], predicted_point[0]], 
            [last_point[1], predicted_point[1]], 
            [last_point[2], predicted_point[2]], 
            color='purple', linestyle='--', label='Predicted Path')
    
    ax.plot([last_point[0], target_point[0]], 
            [last_point[1], target_point[1]], 
            [last_point[2], target_point[2]], 
            color='orange', linestyle=':', label='Target Path')

    ax.scatter(predicted_point[0], predicted_point[1], predicted_point[2], 
               color='purple', marker='s', s=100, label='Predicted Next Pose')
    ax.scatter(target_point[0], target_point[1], target_point[2], 
               color='orange', marker='*', s=100, label='Target Next Pose')
    
    ax.set_title('3D Trajectory Prediction')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    return fig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        forward pass of the model
        """
        return self.pe[:x.size(1), :]

class PoseTransformerLightning(pl.LightningModule):
    def __init__(
        self, 
        pose_dim=7, 
        d_model=256, 
        nhead=8,     
        num_encoder_layers=2,
        dropout=0.1,
        max_seq_len=20,
        input_channels=3
    ):
        super().__init__()
        
       # Image Feature Extraction Layers
        self.image_features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Image Feature Projection
        self.image_feature_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, d_model),
            nn.ReLU()
        )
        
        # Pose Embedding Layer
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Pose Prediction Head
        self.pose_prediction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, pose_dim)
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters()
        self.test_data = {
            "poses": [],
            "predicted_poses": [],
            "target_poses" : [],
            "loss": []
        }
    
    def forward(self, images, poses):
        """
        Forward pass for sequence-to-point pose prediction
        """
        batch_size, seq_len, c, h, w = images.shape
        

        image_features = []
        for t in range(seq_len):
            img_feat = self.image_features(images[:, t, :, :, :])
            projected_img_feat = self.image_feature_projector(img_feat)
            image_features.append(projected_img_feat)
        image_features = torch.stack(image_features, dim=1)  # [batch_size, seq_len, d_model]
        
        # Embed poses
        pose_features = self.pose_embedding(poses)
        

        combined_features = image_features + pose_features

        pos_encoded_features = combined_features + self.positional_encoding(combined_features)

        pos_encoded_features = self.dropout(pos_encoded_features)
        

        pos_encoded_features = pos_encoded_features.permute(1, 0, 2)
        

        encoded_features = self.transformer_encoder(pos_encoded_features)
        

        last_timestep_features = encoded_features[-1]

        predicted_pose = self.pose_prediction_head(last_timestep_features)
        
        return predicted_pose

    

    def quaternion_loss(self, q_pred, q_true):
        """
        Compute the loss between predicted and true quaternions using dot product.
        """
        dot_product = torch.sum(q_pred * q_true, dim=-1)  
        dot_product = torch.clamp(dot_product, -1.0, 1.0) 
        angle = torch.acos(dot_product) 
        return torch.mean(angle ** 2)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning

        """
        images, poses, target_poses = batch
        

        predicted_poses = self(images, poses) 
        
        predicted_poses = self(images, poses) 
        
        predicted_positions = predicted_poses[:, :3]*100  #convert to cm
        predicted_quaternions = predicted_poses[:, 3:] 
        
        target_positions = target_poses[:, :3]*100 
        target_quaternions = target_poses[:, 3:]  
        
       
        position_loss = self.loss_fn(predicted_positions, target_positions)

        position_loss=position_loss*100
        
       
        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)
        
        # Total loss
        total_loss = position_loss + quaternion_loss

        total_loss=total_loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with trajectory visualization
        """
        images, poses, target_poses = batch
        

        predicted_poses = self(images, poses)
        

        loss = self.loss_fn(predicted_poses, target_poses)

        predicted_positions = predicted_poses[:, :3]*100 
        predicted_quaternions = predicted_poses[:, 3:]  
        
        target_positions = target_poses[:, :3]*100  
        target_quaternions = target_poses[:, 3:]  
        
        position_loss = self.loss_fn(predicted_positions, target_positions)

        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)
        
        total_val_loss = position_loss + quaternion_loss

        self.log('val_loss', total_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx < 3:  
            prediction_horizon = images.shape[1]

            fig = plot_trajectory(
                poses, 
                predicted_poses.unsqueeze(0), 
                target_poses.unsqueeze(0), 
                prediction_horizon
            )
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            
            img = mpimg.imread(buf)

            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_image(
                f'trajectory_prediction/batch_{batch_idx}', 
                img.transpose(2, 0, 1), 
                global_step=self.global_step
            )
        
        return total_val_loss
    
    def test_step(self, batch, batc_idx):
        images, poses, target_poses = batch
        predicted_poses = self(images, poses)

        predicted_positions = predicted_poses[:, :3]*100  
        predicted_quaternions = predicted_poses[:, 3:]  
        
        target_positions = target_poses[:, :3]*100  
        target_quaternions = target_poses[:, 3:]  

        position_loss = self.loss_fn(predicted_positions, target_positions)

        quaternion_loss = self.quaternion_loss(predicted_quaternions, target_quaternions)

        total_test_loss = position_loss + quaternion_loss
        
        self.test_data["poses"].append(poses.detach().cpu())
        self.test_data["predicted_poses"].append(predicted_poses.detach().cpu())
        self.test_data["target_poses"].append(target_poses.detach().cpu())
        self.test_data["loss"].append(total_test_loss)
        return total_test_loss
    
    def on_test_epoch_end(self):
        all_true_poses = torch.cat(self.test_data["poses"])
        all_pred_samples = torch.cat(self.test_data["predicted_poses"])
        all_targets = torch.cat(self.test_data["target_poses"])
        avg_test_loss = torch.mean(torch.tensor(self.test_data["loss"]))
        self.log('test_loss', avg_test_loss, prog_bar=True)

        for i in range(len(all_true_poses)):
            fig = plot_trajectory(
                all_true_poses[i:i+1], 
                all_pred_samples[i:i+1], 
                all_targets[i:i+1]
            )
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)

            img = mpimg.imread(buf)
            self.logger.experiment.add_image(
                f'test_trajectory_{i}', 
                img.transpose(2, 0, 1),
                global_step=i
            )

            plt.close(fig)

    def configure_optimizers(self):
        """
        Configure optimizer for training
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3,  
            weight_decay=1e-5 
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
