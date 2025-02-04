import torch

def generate_block_matrix(batch_vector):
    # Mark the positions of padded elements (-1)
    is_padded = batch_vector == -1

    # Determine change points for each sequence in the batch, ignoring transitions to padding
    change_mask = torch.cat([torch.ones_like(batch_vector[:, :1], dtype=torch.bool), batch_vector[:, :-1] != batch_vector[:, 1:]], dim=1)
    change_mask[is_padded] = False

    # Get the starting indices of each block for each sequence in the batch
    boundaries = change_mask.nonzero(as_tuple=False)
    batch_boundaries_list = [boundaries[boundaries[:,0] == i][:,1].tolist() for i in range(batch_vector.shape[0])]

    # Find the maximum number of boundaries across all sequences, excluding padding
    max_boundaries = max(len(b) for b in batch_boundaries_list)

    # Pad each boundary list to the maximum length and construct the mask
    padded_boundaries = torch.full((len(batch_boundaries_list), max_boundaries), -1)
    boundary_mask = torch.zeros_like(padded_boundaries).bool()

    for i, boundary in enumerate(batch_boundaries_list):
        padded_boundaries[i, :len(boundary)] = torch.tensor(boundary)
        boundary_mask[i, :len(boundary)] = True

    # Expand dimensions for broadcasting
    vec_exp = batch_vector.unsqueeze(2)
    vec_transposed = batch_vector.unsqueeze(1)

    same_value_mask = (vec_exp == vec_transposed)
    block_matrix_mask = (same_value_mask).float()

    block_matrix_mask = (block_matrix_mask - 1) * 1e9

    return block_matrix_mask, padded_boundaries, boundary_mask

def fetch_sentence_embeddings(values, indices):
	batch_size, M = indices.shape
	batch_indices = torch.arange(batch_size)[:, None].expand(-1, M)
	fetched_values = values[batch_indices, indices]
	return fetched_values 

def inverse_fetch(values, indices, mask, N):
	batch_size, M, D = values.shape
	output = torch.zeros(batch_size, N, D, device=values.device, dtype=values.dtype)
	batch_indices = torch.arange(batch_size)[:, None].expand(-1, M).to(indices.device)
	output[batch_indices[mask], indices[mask]] = values[mask]

	return output
