#!/usr/bin/env python3
# convert_ci_to_aabb_indices.py

def read_occupancy_tokens(path):
    """
    Read whitespace‐separated tokens from the file.
    Expected tokens per orbital: '2', 'a', 'b', or '0'.
    Returns a list of tokens.
    """
    with open(path, 'r') as f:
        return f.read().split()

def tokens_to_abab_bits(tokens):
    """
    Given a list of N tokens, produce a flat list of 2N bits
    in "abab" order: [α₁, β₁, α₂, β₂, ..., α_N, β_N].
    Token mapping:
      '2' → (1,1)
      'a' → (1,0)
      'b' → (0,1)
      '0' → (0,0)
    """
    bits = []
    for tok in tokens:
        if tok == '2':
            bits.extend([1, 1])
        elif tok == 'a':
            bits.extend([1, 0])
        elif tok == 'b':
            bits.extend([0, 1])
        elif tok == '0':
            bits.extend([0, 0])
        else:
            raise ValueError(f"Unknown token {tok!r}")
    return bits

def extract_indices(abab_bits):
    """
    From a 2N-length abab_bits list, return two lists of 1-based positions:
      alpha_positions = [ i | i odd and abab_bits[i-1] == 1 ]
      beta_positions  = [ i | i even and abab_bits[i-1] == 1 ]
    """
    alpha_positions = []
    beta_positions  = []
    for pos, bit in enumerate(abab_bits, start=1):
        if bit == 1:
            if pos % 2 == 1:
                alpha_positions.append(pos)
            else:
                beta_positions.append(pos)
    return alpha_positions, beta_positions

def main():
    tokens = read_occupancy_tokens("ci.txt")
    abab_bits = tokens_to_abab_bits(tokens)
    alpha_pos, beta_pos = extract_indices(abab_bits)

    # Format as "aabb": all alpha indices (with 'a'), then all beta indices (with 'b')
    electron_labels = [f"{i}" for i in alpha_pos] + [f"{i}" for i in beta_pos]
    print(" ".join(electron_labels))

if __name__ == "__main__":
    main()

