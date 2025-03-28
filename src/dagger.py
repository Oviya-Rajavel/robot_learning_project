import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks.taskboard_robothon import TaskboardRobothon
# from behaviour_cloning_transformer import PoseTransformerLightning
from behaviour_cloning_dataloader import create_dataloaders
from sequence_dataloader import SequencePredictionDataset, create_sequence_prediction_dataloaders
# from behaviour_clonning_sequence import PoseTransformerLightning
from final_behaviour_cloning import PoseTransformerLightning

class ImprovedImitationLearning:
    def __init__(self, 
                 input_channels=3, 
                 window_size=20,
                 pose_input_dim=7, 
                 pose_output_dim=7,
                 learning_rate=1e-4,
                 model_path=None,
                 validation_split=0.2):
        """
        Improved Imitation Learning class with DAgger 
        """
        self.model = PoseTransformerLightning(
            input_channels=input_channels,
            pose_dim=pose_input_dim,
            max_seq_len=window_size
        )
        
        # Load pre-trained weights if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate
        )

        self.window_size = window_size
        self.input_channels = input_channels
        self.validation_split = validation_split
    
    def prepare_sequence_data(self, demos):
        """
        Prepare sequence data for DAgger training
        """
        all_images = []
        all_poses = []
        all_actions = []
        
        for demo in demos:
           
            images = [obs.wrist_rgb for obs in demo._observations]
            poses = [obs.gripper_pose for obs in demo._observations]
            actions = [obs.gripper_pose for obs in demo._observations[1:]] + [poses[-1]]
            
            for start in range(0, len(images) - self.window_size + 1):
                window_images = images[start:start+self.window_size]
                window_poses = poses[start:start+self.window_size]
                window_action = actions[start+self.window_size-1]
                
                all_images.append(window_images)
                all_poses.append(window_poses)
                all_actions.append(window_action)
        
        return (
            np.array(all_images),
            np.array(all_poses),
            np.array(all_actions)
        )
    
    def split_data(self, all_images, all_poses, all_actions):
        """
        Split data into training and validation sets
        """
       
        total_samples = len(all_images)
        val_samples = int(total_samples * self.validation_split)
        train_samples = total_samples - val_samples
        

        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:]

        train_images = all_images[train_indices]
        train_poses = all_poses[train_indices]
        train_actions = all_actions[train_indices]
        
        val_images = all_images[val_indices]
        val_poses = all_poses[val_indices]
        val_actions = all_actions[val_indices]
        
        return (
            train_images, train_poses, train_actions,
            val_images, val_poses, val_actions
        )
    
    def dagger_training(self, demos, num_iterations=10, expert_ratio=0.5, batch_size=32):
        """
        Improved DAgger-like training approach with validation
        """

        all_images, all_poses, all_actions = self.prepare_sequence_data(demos)
        
        # Split data into training and validation sets
        (train_images, train_poses, train_actions,
         val_images, val_poses, val_actions) = self.split_data(
            all_images, all_poses, all_actions
        )
        
        for iteration in range(num_iterations):
            print(f"Training Iteration {iteration + 1}/{num_iterations}")

            self.model.train()
            train_loss = self._train_epoch(
                train_images, train_poses, train_actions, 
                batch_size, expert_ratio
            )
            

            self.model.eval()
            val_loss = self._validate(
                val_images, val_poses, val_actions, 
                batch_size, expert_ratio
            )
            
            print(f"Iteration {iteration + 1}: "
                  f"Train Loss = {train_loss:.4f}, "
                  f"Validation Loss = {val_loss:.4f}")
        
        # Save the  model
        torch.save(self.model.state_dict(), 'dagger_trained_pose_transformer_with_val.pt')
        
        return self.model
    
    def _train_epoch(self, images, poses, actions, batch_size, expert_ratio):
        """
        Perform a training epoch
        """
        total_loss = 0.0
        num_batches = len(images) // batch_size + (1 if len(images) % batch_size else 0)
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(images))
            

            batch_images = torch.FloatTensor(images[start:end]) / 255.0
            batch_images = batch_images.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
            
            batch_poses = torch.FloatTensor(poses[start:end])
            batch_actions = torch.FloatTensor(actions[start:end])
            

            with torch.no_grad():
                predicted_actions = self.model(batch_images, batch_poses)
            
            # Interpolate between expert and model actions
            interpolated_actions = (
                expert_ratio * batch_actions + 
                (1 - expert_ratio) * predicted_actions
            )
            
            
            self.optimizer.zero_grad()
            
            
            output = self.model(batch_images, batch_poses)
            
            # Compute loss
            loss = self.model.loss_fn(output, interpolated_actions)
            

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _validate(self, images, poses, actions, batch_size, expert_ratio):
        """
        Perform validation
        """
        total_loss = 0.0
        num_batches = len(images) // batch_size + (1 if len(images) % batch_size else 0)
        
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(images))
                
                batch_images = torch.FloatTensor(images[start:end]) / 255.0
                batch_images = batch_images.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
                
                batch_poses = torch.FloatTensor(poses[start:end])
                batch_actions = torch.FloatTensor(actions[start:end])
                

                predicted_actions = self.model(batch_images, batch_poses)
                
                # Interpolate 
                interpolated_actions = (
                    expert_ratio * batch_actions + 
                    (1 - expert_ratio) * predicted_actions
                )
                
                # loss
                loss = self.model.loss_fn(predicted_actions, batch_actions)
                
                total_loss += loss.item()
        
        return total_loss / num_batches

def main():

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), 
            gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        headless=False
    )
    env.launch()
    
    # Get task
    task = env.get_task(TaskboardRobothon)

    demos = task.get_demos(10, live_demos=True)
    

    il = ImprovedImitationLearning(
        input_channels=3, 
        window_size=20,  
        pose_input_dim=7, 
        pose_output_dim=7,
        learning_rate=1e-4,
        model_path='/media/sun/Expansion/Robot_learning/rlbench_transformer.pth',
        validation_split=0.2 
    )
    

    trained_model = il.dagger_training(
        demos, 
        num_iterations=10, 
        expert_ratio=0.5,   
        batch_size=32      
    )

    env.shutdown()

if __name__ == "__main__":
    main()