#!/usr/bin/env python3
"""
Process a ci.txt file where each line has:
  <index> <coefficient> <token1> ... <tokenN>
with tokens in {'2','a','b','0'}. For each line, this script:
  1. Reads index (ignored), coefficient, and N occupancy tokens.
  2. Converts tokens to α/β bit‐lists of length N.
  3. Writes to alpha_ci.txt and beta_ci.txt lines of the form:
     String: <bitstring>, Coeffs: <coefficient>
"""

def read_records(path):
    """Read nonempty lines from ci.txt, split into fields."""
    records = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            records.append(parts)
    return records

def tokens_to_bits(tokens):
    """Map occupancy tokens to parallel alpha/beta bit‐lists."""
    alpha = []
    beta  = []
    for tok in tokens:
        if tok == '2':
            alpha.append('1'); beta.append('1')
        elif tok == 'a':
            alpha.append('1'); beta.append('0')
        elif tok == 'b':
            alpha.append('0'); beta.append('1')
        elif tok == '0':
            alpha.append('0'); beta.append('0')
        else:
            raise ValueError(f"Unknown token {tok!r}")
    return "".join(alpha), "".join(beta)

def main():
    records = read_records("shci_results.txt")
    if not records:
        print("No valid records found in ci.txt.")
        return

    with open("alpha_ci.txt", "w") as fa, open("beta_ci.txt", "w") as fb:
        for parts in records:
            _, coeff, *tokens = parts
            alpha_str, beta_str = tokens_to_bits(tokens)

            fa.write(f"String: {alpha_str}, Coeffs: {coeff}\n")
            fb.write(f"String: {beta_str}, Coeffs: {coeff}\n")

    print(f"Wrote {len(records)} lines to alpha_ci.txt and beta_ci.txt")

if __name__ == "__main__":
    main()


