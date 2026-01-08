/**
 * Atom3D Flood Fill CUDA Kernels
 * 
 * Self-contained implementation for 3D flood fill using 26-neighbor connectivity.
 * No external dependencies required.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel: Flood Fill using 26-neighbor connectivity
__global__ void flood_fill_kernel(
    const bool* occupancy,
    int* mask,
    const int* current_frontier,
    int frontier_size,
    int* next_frontier,
    int* next_frontier_size,
    int D, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int voxel_idx = current_frontier[idx];
    int z = voxel_idx / (H * W);
    int y = (voxel_idx % (H * W)) / W;
    int x = voxel_idx % W;
    
    bool has_collision = false;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D)
                    continue;

                int n_idx = nz * (H * W) + ny * W + nx;

                if (occupancy[n_idx]) {
                    has_collision = true;
                }

                if (atomicCAS(&mask[n_idx], -1, -1) == -1) {
                    if (!occupancy[n_idx]) {
                        int old_val = atomicCAS(&mask[n_idx], -1, 1);
                        if (old_val == -1) {
                            int pos = atomicAdd(next_frontier_size, 1);
                            next_frontier[pos] = n_idx;
                        }
                    }
                }
            }
        }
    }
    
    if (has_collision) {
        mask[voxel_idx] = 0;
    } else if (mask[voxel_idx] == -1) {
        mask[voxel_idx] = 1;
    }
}

// Host wrapper for CUDA kernel
void flood_fill_cuda(
    torch::Tensor occupancy,
    torch::Tensor mask,
    torch::Tensor current_frontier,
    int frontier_size,
    torch::Tensor next_frontier,
    torch::Tensor next_frontier_size,
    int D, int H, int W
) {
    const int threads = THREADS_PER_BLOCK;
    const int blocks = (frontier_size + threads - 1) / threads;

    flood_fill_kernel<<<blocks, threads>>>(
        occupancy.data_ptr<bool>(),
        mask.data_ptr<int>(),
        current_frontier.data_ptr<int>(),
        frontier_size,
        next_frontier.data_ptr<int>(),
        next_frontier_size.data_ptr<int>(),
        D, H, W
    );
    cudaDeviceSynchronize();
}

// Main flood fill function
torch::Tensor flood_fill(torch::Tensor occupancy) {
    occupancy = occupancy.contiguous();
    auto sizes = occupancy.sizes();
    int D = sizes[0];
    int H = sizes[1];
    int W = sizes[2];
    int num_voxels = D * H * W;

    auto mask = torch::full({D, H, W}, -1, torch::TensorOptions().device(occupancy.device()).dtype(torch::kInt32));

    auto options = torch::TensorOptions().device(occupancy.device()).dtype(torch::kInt32);
    auto current_frontier = torch::empty({num_voxels}, options);
    auto next_frontier = torch::empty({num_voxels}, options);
    auto next_frontier_size = torch::zeros({1}, options);

    int frontier_size = 0;

    int h_one = 1;
    cudaMemcpy(mask.data_ptr<int>(), &h_one, sizeof(int), cudaMemcpyHostToDevice);
    int h_zero = 0;
    cudaMemcpy(current_frontier.data_ptr<int>(), &h_zero, sizeof(int), cudaMemcpyHostToDevice);
    frontier_size = 1; 

    while (frontier_size > 0) {
        next_frontier_size.fill_(0);

        flood_fill_cuda(
            occupancy,
            mask,
            current_frontier,
            frontier_size,
            next_frontier,
            next_frontier_size,
            D, H, W
        );

        frontier_size = next_frontier_size.item<int>();

        auto temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
    }

    return mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flood_fill", &flood_fill, "3D Flood Fill (Atom3D built-in)");
}
