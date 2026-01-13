/**
 * Atom3d Flood Fill CUDA Kernels
 * 
 * Self-contained implementation for 3D flood fill using 26-neighbor connectivity.
 * No external dependencies required.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

// CUDA kernel: Flood Fill using variable connectivity
// Semantics:
//   mask == -2: Dry (Unreachable)
//   mask == -1: Dam (Occupied voxel reached by flood)
//   mask ==  1: Collision (Water voxel adjacent to Dam)
//   mask ==  2: Water (Pure Water voxel)
__global__ void flood_fill_kernel(
    const bool* occupancy,
    int* mask,
    const int* current_frontier,
    int frontier_size,
    int* next_frontier,
    int* next_frontier_size,
    int D, int H, int W,
    int connectivity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int voxel_idx = current_frontier[idx];
    int z = voxel_idx / (H * W);
    int y = (voxel_idx % (H * W)) / W;
    int x = voxel_idx % W;
    
    bool is_collision = false;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                
                // Check connectivity
                int dist = abs(dx) + abs(dy) + abs(dz);
                if (connectivity == 6 && dist > 1) continue;
                if (connectivity == 18 && dist > 2) continue;

                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D)
                    continue;

                int n_idx = nz * (H * W) + ny * W + nx;

                if (occupancy[n_idx]) {
                    // Neighbor is Occupied -> Mark as Dam (-1)
                    // We try to switch from Dry (-2) to Dam (-1).
                    // If it was already Dam (-1), atomicCAS returns -1.
                    atomicCAS(&mask[n_idx], -2, -1);
                    
                    // Current voxel touches a Dam -> It IS a Collision
                    is_collision = true;
                } else {
                    // Neighbor is Empty -> Try to fill as Water (2)
                    // Only fill if it is currently Dry (-2)
                    int old_val = atomicCAS(&mask[n_idx], -2, 2);
                    if (old_val == -2) {
                        // Successfully claimed as Water. Add to next frontier.
                        int pos = atomicAdd(next_frontier_size, 1);
                        next_frontier[pos] = n_idx;
                    }
                }
            }
        }
    }
    
    // If we found any collision, upgrade current voxel from Water (2) to Collision (1)
    if (is_collision) {
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
    int D, int H, int W,
    int connectivity
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
        D, H, W,
        connectivity
    );
    cudaDeviceSynchronize();
}

// Main flood fill function
torch::Tensor flood_fill(
    torch::Tensor occupancy,
    int start_x, int start_y, int start_z,
    int connectivity
) {
    occupancy = occupancy.contiguous();
    auto sizes = occupancy.sizes();
    int D = sizes[0];
    int H = sizes[1];
    int W = sizes[2];
    int num_voxels = D * H * W;

    // Initialize mask with -2 (Dry)
    auto mask = torch::full({D, H, W}, -2, torch::TensorOptions().device(occupancy.device()).dtype(torch::kInt32));

    auto options = torch::TensorOptions().device(occupancy.device()).dtype(torch::kInt32);
    auto current_frontier = torch::empty({num_voxels}, options);
    auto next_frontier = torch::empty({num_voxels}, options);
    auto next_frontier_size = torch::zeros({1}, options);

    // Setup start point
    if (start_x >= 0 && start_x < W && 
        start_y >= 0 && start_y < H && 
        start_z >= 0 && start_z < D) {
        
        int start_idx = start_z * (H * W) + start_y * W + start_x;
        
        // Mark start as Water (2)
        int initial_val = 2;
        cudaMemcpy(mask.data_ptr<int>() + start_idx, &initial_val, sizeof(int), cudaMemcpyHostToDevice);
        
        // Add start to frontier
        cudaMemcpy(current_frontier.data_ptr<int>(), &start_idx, sizeof(int), cudaMemcpyHostToDevice);
        int frontier_size = 1;

        while (frontier_size > 0) {
            next_frontier_size.fill_(0);

            flood_fill_cuda(
                occupancy,
                mask,
                current_frontier,
                frontier_size,
                next_frontier,
                next_frontier_size,
                D, H, W,
                connectivity
            );

            frontier_size = next_frontier_size.item<int>();

            auto temp = current_frontier;
            current_frontier = next_frontier;
            next_frontier = temp;
        }
    }

    return mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flood_fill", &flood_fill, "3D Flood Fill (Atom3D built-in)");
}
