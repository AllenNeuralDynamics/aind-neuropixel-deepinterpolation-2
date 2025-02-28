import torch
import torch.nn as nn
import numpy as np
import zarr
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing as mp
from multiprocessing import Pool

# Network Library with dense layer, no attention
class InterpolationNetworkLibrary:
    @staticmethod
    def sparse_unet_2d(in_channels, out_channels=1, grid_height=4, grid_width=96):
        class SparseUNet2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_proj = nn.Linear(384, grid_height * grid_width)
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
                x_2d = latent.view(batch_size, in_channels, self.grid_height, self.grid_width)
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
                return out.squeeze(1)
        
        return SparseUNet2D()

# Function to transfer Zarr to memory-mapped file in parallel
def transfer_zarr_to_memmap(zarr_data, memmap_path, num_processes=4):
    shape = zarr_data.shape
    dtype = zarr_data.dtype
    memmap_data = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=shape)
    
    def load_chunk(chunk_idx):
        start = chunk_idx * (shape[0] // num_processes)
        end = (chunk_idx + 1) * (shape[0] // num_processes) if chunk_idx < num_processes - 1 else shape[0]
        memmap_data[start:end] = zarr_data[start:end]
        print(f"Transferred chunk {chunk_idx}: frames {start} to {end}")
    
    with Pool(processes=num_processes) as pool:
        pool.map(load_chunk, range(num_processes))
    
    return memmap_data

# Custom Dataset using memory-mapped file
class MemMapInterpolationDataset(Dataset):
    def __init__(self, memmap_path, shape, dtype, n_frames=2, subset_size=10000, sample_size=10):
        self.memmap_data = np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)
        self.n_frames = n_frames
        self.total_frames = shape[0]
        self.subset_size = min(subset_size, self.total_frames - 2 * n_frames)
        self.valid_indices = np.random.choice(
            np.arange(n_frames, self.total_frames - n_frames),
            size=self.subset_size,
            replace=False
        )
        print(f"Selected {len(self.valid_indices)} frames for training")
        
        print("Computing normalization stats...")
        sample_indices = np.random.choice(self.total_frames, min(sample_size, self.total_frames), replace=False)
        sample_data = self.memmap_data[sample_indices]
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
                frame = self.memmap_data[center_idx + j]
                input_frames.append(frame)
        
        target_frame = self.memmap_data[center_idx]
        
        input_frames_np = np.stack(input_frames)
        norm_input_frames = (input_frames_np - self.global_mean) / self.global_std
        norm_target_frame = (target_frame - self.global_mean) / self.global_std
        
        return (torch.FloatTensor(norm_input_frames), 
                torch.FloatTensor(norm_target_frame))

# Training function
def train_model(model, dataloader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)-1}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')

# Inference function
def denoise_frame(model, memmap_data, frame_idx, n_frames, global_mean, global_std):
    model.eval()
    with torch.no_grad():
        input_frames = []
        for j in range(-n_frames, n_frames + 1):
            if j != 0:
                frame = memmap_data[frame_idx + j]
                input_frames.append(frame)
        input_frames_np = np.stack(input_frames)
        
        norm_input_frames = (input_frames_np - global_mean) / global_std
        
        input_stack = torch.FloatTensor(norm_input_frames).unsqueeze(0).to(device)
        denoised = model(input_stack)
        denoised = denoised * global_std + global_mean
        return denoised.squeeze().cpu().numpy()

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    
    # Configuration
    N_FRAMES = 2
    BATCH_SIZE = 128
    SUBSET_SIZE = 10000
    SAMPLE_SIZE = 10
    NUM_WORKERS = 8  # Adjust based on CPU cores
    ZARR_PATH = "/data/ecephys_660948_2023-05-01_20-02-08/ecephys_compressed/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA.zarr"
    ARRAY_NAME = 'traces_seg0'
    MEMMAP_PATH = "/data/memmap_data.dat"  # Local path for memory-mapped file
    NUM_PROCESSES = 10  # For parallel transfer
    
    # Step 1: Open Zarr and get array info
    zarr_group = zarr.open(ZARR_PATH, mode='r')
    zarr_data = zarr_group[ARRAY_NAME]
    shape = zarr_data.shape
    dtype = zarr_data.dtype
    print(f"Zarr data shape: {shape}, dtype: {dtype}")
    
    # Step 2: Transfer Zarr to memory-mapped file in parallel
    if not os.path.exists(MEMMAP_PATH):
        print("Transferring Zarr to memory-mapped file...")
        memmap_data = transfer_zarr_to_memmap(zarr_data, MEMMAP_PATH, num_processes=NUM_PROCESSES)
    else:
        print("Using existing memory-mapped file...")
        memmap_data = np.memmap(MEMMAP_PATH, dtype=dtype, mode='r', shape=shape)
    
    # Step 3: Use memory-mapped file in dataset
    dataset = MemMapInterpolationDataset(
        MEMMAP_PATH, 
        shape=shape, 
        dtype=dtype, 
        n_frames=N_FRAMES, 
        subset_size=SUBSET_SIZE, 
        sample_size=SAMPLE_SIZE
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    net_library = InterpolationNetworkLibrary()
    model = net_library.sparse_unet_2d(in_channels=2*N_FRAMES, grid_height=4, grid_width=96).to(device)
    
    train_model(model, dataloader)
    
    # For inference, use the memory-mapped data
    test_frame_idx = 50
    denoised_frame = denoise_frame(model, memmap_data, test_frame_idx, N_FRAMES, 
                                   dataset.global_mean, dataset.global_std)
    print(f"Denoised frame shape: {denoised_frame.shape}")


# Main execution
if __name__ == "__main__":
    try:
        numcodecs.wavpack
    except AttributeError:
        print("Warning: 'wavpack' codec not available. Install with 'pip install numcodecs[wavpack]' if needed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    N_FRAMES = 2
    BATCH_SIZE = 128
    SUBSET_SIZE = 10000
    CHUNK_SIZE = 30000
    SAMPLE_SIZE = 10
    NUM_WORKERS = 2
    ZARR_PATH = "/data/ecephys_660948_2023-05-01_20-02-08/ecephys_compressed/experiment1_Record Node 104#Neuropix-PXI-100.ProbeA.zarr"
    ARRAY_NAME = 'traces_seg0'
    
    # If the zarr file doesn't exist, create fake data
    if not os.path.exists(ZARR_PATH):
        print("Creating fake zarr data...")
        create_fake_zarr_data(ZARR_PATH)
    else:
        print("Using existing zarr data...")
    
    # Open the zarr group to check its contents
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

    # Set these to test in-memory loading. 
    # Optionally, specify a memory_range (start, end) to load only a portion.
    LOAD_TO_MEMORY = True
    
    dataset = ZarrInterpolationDataset(
        ZARR_PATH, 
        array_name=ARRAY_NAME, 
        n_frames=N_FRAMES, 
        chunk_size=CHUNK_SIZE, 
        sample_size=SAMPLE_SIZE, 
        subset_size=SUBSET_SIZE,
        load_to_memory=LOAD_TO_MEMORY,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    net_library = InterpolationNetworkLibrary()
    model = net_library.sparse_unet_2d(in_channels=2*N_FRAMES, grid_height=4, grid_width=96).to(device)
    
    train_model(model, dataloader)
    
    # For inference, you can also use the in-memory data from the dataset
    # Here we use dataset.zarr_data which is now in memory if LOAD_TO_MEMORY is True
    test_frame_idx = 50
    denoised_frame = denoise_frame(model, dataset.zarr_data, test_frame_idx, N_FRAMES, 
                                   dataset.global_mean, dataset.global_std)
    
    print(f"Denoised frame shape: {denoised_frame.shape}")
