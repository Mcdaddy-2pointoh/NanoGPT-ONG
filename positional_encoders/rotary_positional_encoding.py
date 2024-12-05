import torch


def RotaryPositionEncoding(q, k, T, attention_head_size, device):
    """
    Apply Rotary Positional Embedding (RoPE) to q and k tensors.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, time, attention_head_size)
        k (torch.Tensor): Key tensor of shape (batch_size, time, attention_head_size)
        T (int): Sequence length
        attention_head_size (int): Dimension of each attention head
        device (str): Device ('cuda' or 'cpu')

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: RoPE-applied q and k
    """

    # Generate position indices (0, 1, 2, ..., T-1) for each token in the sequence.
    # Shape: (T, 1)
    position_ids = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)

    # Calculate inverse frequencies for the sinusoidal embeddings.
    # head_dim/2 frequencies are computed (only for even positions in embeddings).
    # Shape: (head_dim // 2,)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, attention_head_size, 2, device=device).float() / attention_head_size))

    # Compute the product of position indices and inverse frequencies using Einstein summation.
    # This creates a sinusoidal input for each position and frequency.
    # Shape: (T, head_dim // 2)
    sinusoidal_inp = torch.einsum("i,j->ij", position_ids.squeeze(1), inv_freq)

    # Calculate the sine of the sinusoidal input for each position and frequency.
    # Shape: (1, T, head_dim // 2), unsqueezed to match batch dimensions later.
    sin = sinusoidal_inp.sin().unsqueeze(0)


    # Calculate the cosine of the sinusoidal input for each position and frequency.
    # Shape: (1, T, head_dim // 2), unsqueezed similarly to `sin`.
    cos = sinusoidal_inp.cos().unsqueeze(0)

    # Split the query tensor `q` into even and odd indexed positions along the last dimension.
    # q1 contains values at even indices, q2 contains values at odd indices.
    # Shape for q1 and q2: (batch_size, T, head_dim // 2)
    q1, q2 = q[..., ::2], q[..., 1::2]

    # Similarly, split the key tensor `k` into even and odd indexed positions.
    # Shape for k1 and k2: (batch_size, T, head_dim // 2)
    k1, k2 = k[..., ::2], k[..., 1::2]

    # Apply rotation to q1 and q2 using sine and cosine.
    # q_rot = [q1 * cos - q2 * sin, q1 * sin + q2 * cos]
    # Shape: (batch_size, T, head_dim)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)

    # Apply the same rotation logic to k1 and k2.
    # k_rot = [k1 * cos - k2 * sin, k1 * sin + k2 * cos]
    # Shape: (batch_size, T, head_dim)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    # Return the rotated query and key tensors, embedding positional information.
    return q_rot, k_rot
