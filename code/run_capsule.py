import torch
import torch.nn as nn
import numpy as np
import zarr
from torch.utils.data import Dataset, DataLoader
import os
import numcodecs

# Network Library (unchanged)
class InterpolationNetworkLibrary:
    @staticmethod
    def sparse_unet_2d(in_channels, out_channels=1, latent_size=1024, grid_height=4, grid_width=192):
        class SparseUNet2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_proj = nn.Linear(384, latent_size)
                self.pos_attention = nn.Sequential(
                    nn.Linear(latent_size, latent_size),
                    nn.ReLU(),
                    nn.Linear(latent_size, grid_height * grid_width),
                    nn.Softmax(dim=-1)
                )
                self.conv1 = nn.Conv2d(in_channels, 64, (3, 3), padding=1)
                self.pool1 = nn.MaxPool2d((2, 2))
                self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
                self.pool2 = nn.MaxPool2d((1, 2))
                self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
                self.pool3 = nn.MaxPool2d((1, 2))
                self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)
                self.conv5 = nn.Conv2d(512, 1024, (3, 3), padding=1)
                self.up1 = nn.Upsample(scale_factor=(1, 2), mode='bilinear')
                self.conv7 = nn.Conv2d(1024 + 256, 512, (3, 3), padding=1)
                self.up2 = nn.Upsample(scale_factor=(1, 2), mode='bilinear')
                self.conv8 = nn.Conv2d(512 + 128, 256, (3, 3), padding=1)
                self.up3 = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
                self.conv9 = nn.Conv2d(256 + 64, 128, (3, 3), padding=1)
                self.conv10 = nn.Conv2d(128, 64, (3, 3), padding=1)
                self.decoded = nn.Conv2d(64, out_channels, (1, 1), padding=0)
                self.final_dense = nn.Linear(grid_height * grid_width, 384)
                self.relu = nn.ReLU()
                self.grid_height = grid_height
                self.grid_width = grid_width
                
            def forward(self, x):
                batch_size = x.size(0)
                latent = self.relu(self.latent_proj(x))
                pos_scores = self.pos_attention(latent.mean(dim=1))
                x_2d = torch.zeros(batch_size, in_channels, self.grid_height, self.grid_width, 
                                 device=x.device)
                for i in range(batch_size):
                    grid_flat = torch.einsum('cl,g->g', latent[i], pos_scores[i])
                    x_2d[i] = grid_flat.view(1, self.grid_height, self.grid_width).expand(in_channels, -1, -1)
                
                conv1 = self.relu(self.conv1(x_2d))
                pool1 = self.pool1(conv1)
                conv2 = self.relu(self.conv2(pool1))
                pool2 = self.pool2(conv2)
                conv3 = self.relu(self.conv3(pool2))
                pool3 = self.pool3(conv3)
                conv4 = self.relu(self.conv4(pool3))
                conv5 = self.relu(self.conv5(conv4))
                up1 = self.up1(conv5)
                conc_up_1 = torch.cat([up1, conv3], dim=1)
                conv7 = self.relu(self.conv7(conc_up_1))
                up2 = self.up2(conv7)
                conc_up_2 = torch.cat([up2, conv2], dim=1)
                conv8 = self.relu(self.conv8(conc_up_2))
                up3 = self.up3(conv8)
                conc_up_3 = torch.cat([up3, conv1], dim=1)
                conv9 = self.relu(self.conv9(conc_up_3))
                conv10 = self.relu(self.conv10(conv9))
                decoded = self.decoded(conv10)
                decoded = decoded.view(batch_size, 1, -1)
                out = self.final_dense(decoded)
                return out.squeeze(1), pos_scores
        
        return SparseUNet2D()

# Function to create fake zarr data (unchanged)
def create_fake_zarr_data(filepath, num_frames=1000, num_channels=384):
    clean_data = np.zeros((num_frames, num_channels), dtype=np.float32)
    for t in range(num_frames):
        clean_data[t] = np.sin(t * 0.1 + np.arange(num_channels) * 0.01)
    
    noisy_data = clean_data + np.random.normal(0, 0.1, clean_data.shape)
    
    zarr_group = zarr.open(filepath, mode='w')
    zarr_store = zarr_group.create_dataset('traces_seg0', shape=noisy_data.shape, 
                                          chunks=(30000, num_channels), dtype='f4')
    zarr_store[:] = noisy_data
    return zarr_store

# Custom Dataset with chunk-aligned subset
class ZarrInterpolationDataset(Dataset):
    def __init__(self, zarr_path, array_name='traces_seg0', n_frames=2, chunk_size=30000, 
                 sample_size=10, subset_size=10000):
        zarr_group = zarr.open(zarr_path, mode='r')
        try:
            self.zarr_data = zarr_group[array_name]
        except KeyError:
            print(f"Available arrays in zarr group: {list(zarr_group.array_keys())}")
            raise KeyError(f"Array '{array_name}' not found in zarr file {zarr_path}")
        except ValueError as e:
            if 'codec not available' in str(e):
                print(f"Codec error: {e}. Please install required codec (e.g., 'pip install numcodecs[wavpack]')")
            raise
        
        self.n_frames = n_frames
        self.chunk_size = chunk_size
        self.total_frames = self.zarr_data.shape[0]
        
        num_chunks = (self.total_frames - 2 * n_frames) // chunk_size
        self.subset_size = min(subset_size, self.total_frames - 2 * n_frames)
        num_subset_chunks = (self.subset_size + chunk_size - 1) // chunk_size
        chunk_indices = np.random.choice(num_chunks, min(num_subset_chunks, num_chunks), replace=False)
        self.valid_indices = []
        for chunk_idx in chunk_indices:
            start = chunk_idx * chunk_size + n_frames
            end = min(start + chunk_size, self.total_frames - n_frames)
            self.valid_indices.extend(range(start, end))
        self.valid_indices = np.array(self.valid_indices[:self.subset_size])
        print(f"Selected {len(self.valid_indices)} frames from {num_subset_chunks} chunks")
        
        print("Computing normalization stats...")
        sample_indices = np.random.choice(self.total_frames, min(sample_size, self.total_frames), replace=False)
        sample_data = np.array([self.zarr_data[i] for i in sample_indices])
        self.global_mean = np.mean(sample_data)
        self.global_std = np.std(sample_data)
        if self.global_std == 0:
            self.global_std = 1.0
        print(f"Global mean: {self.global_mean:.4f}, Global std: {self.global_std:.4f}")
        
    def __len__(self):
        return self.subset_size
    
    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        input_frames = []
        for j in range(-self.n_frames, self.n_frames + 1):
            if j != 0:
                frame = self.zarr_data[center_idx + j]
                input_frames.append(frame)
        
        target_frame = self.zarr_data[center_idx]
        
        input_frames_np = np.stack(input_frames)
        norm_input_frames = (input_frames_np - self.global_mean) / self.global_std
        norm_target_frame = (target_frame - self.global_mean) / self.global_std
        
        return (torch.FloatTensor(norm_input_frames), 
                torch.FloatTensor(norm_target_frame))

# Training function with optimized DataLoader usage
def train_model(model, dataloader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs, pos_scores = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                threshold = 0.01
                active_positions = (pos_scores[0] > threshold).sum().item()
                max_score = pos_scores[0].max().item()
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)-1}, Loss: {loss.item():.4f}")
                print(f"  Active grid positions (> {threshold}): {active_positions}/{model.grid_height * model.grid_width}")
                print(f"  Max position score: {max_score:.4f}")
                
                grid = pos_scores[0].detach().cpu().numpy()
                grid = grid.reshape(model.grid_height, model.grid_width)
                print(f"  Grid ({model.grid_height}x{model.grid_width}, rounded to 3 decimals):")
                print(np.round(grid, 3))
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

# Inference function (unchanged)
def denoise_frame(model, zarr_data, frame_idx, n_frames, global_mean, global_std):
    model.eval()
    with torch.no_grad():
        input_frames = []
        for j in range(-n_frames, n_frames + 1):
            if j != 0:
                frame = zarr_data[frame_idx + j]
                input_frames.append(frame)
        input_frames_np = np.stack(input_frames)
        
        norm_input_frames = (input_frames_np - global_mean) / global_std
        
        input_stack = torch.FloatTensor(norm_input_frames).unsqueeze(0).to(device)
        denoised, _ = model(input_stack)
        denoised = denoised * global_std + global_mean
        return denoised.squeeze().cpu().numpy()

# Main execution
if __name__ == "__main__":
    try:
        numcodecs.wavpack
    except AttributeError:
        print("Warning: 'wavpack' codec not available. Install with 'pip install numcodecs[wavpack]' if needed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N_FRAMES = 2
    BATCH_SIZE = 32  # Increased from 4
    SUBSET_SIZE = 10000
    CHUNK_SIZE = 30000
    SAMPLE_SIZE = 10
    NUM_WORKERS = 8  # Increased from 2
    ZARR_PATH = "/data/ecephys_660948_2023-05-01_20-02-08/ecephys_compressed/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA.zarr"
    ARRAY_NAME = 'traces_seg0'
    
    if not os.path.exists(ZARR_PATH):
        print("Creating fake zarr data...")
        create_fake_zarr_data(ZARR_PATH)
    else:
        print("Using existing zarr data...")
    
    zarr_group = zarr.open(ZARR_PATH, mode='r')
    print("Zarr group contents:", list(zarr_group.array_keys()))
    try:
        zarr_data = zarr_group[ARRAY_NAME]
        print(f"Data array shape: {zarr_data.shape}")
        print(f"Zarr chunk size: {zarr_data.chunks}")
    except ValueError as e:
        print(f"Error accessing array: {e}")
        print("Likely a codec issue. Please ensure 'wavpack' is installed if required.")
        raise

    dataset = ZarrInterpolationDataset(ZARR_PATH, array_name=ARRAY_NAME, n_frames=N_FRAMES, 
                                     chunk_size=CHUNK_SIZE, sample_size=SAMPLE_SIZE, subset_size=SUBSET_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
                           pin_memory=True if device.type == 'cuda' else False)
    
    net_library = InterpolationNetworkLibrary()
    model = net_library.sparse_unet_2d(in_channels=2*N_FRAMES, latent_size=1024, 
                                     grid_height=4, grid_width=192).to(device)
    
    train_model(model, dataloader)
    
    zarr_data = zarr.open(ZARR_PATH, mode='r')[ARRAY_NAME]
    test_frame_idx = 50
    denoised_frame = denoise_frame(model, zarr_data, test_frame_idx, N_FRAMES, 
                                 dataset.global_mean, dataset.global_std)
    
    print(f"Denoised frame shape: {denoised_frame.shape}")