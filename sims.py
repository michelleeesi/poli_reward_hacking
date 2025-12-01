# %%
import numpy as np


# %%
def utility_function(candidate, voter):
    """
    Computes the utility as the sum of cube roots of elementwise products.
    
    ROBUST VERSION: Handles edge cases that cause NaN
    
    Parameters:
    -----------
    candidate : array-like
        Vector representing candidate position
    voter : array-like
        Vector representing voter position
    epsilon : float, default=1e-10
        Small value to prevent numerical issues
    
    Returns:
    --------
    float
        Sum of cube roots of elementwise products (guaranteed not NaN)
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    voter = np.asarray(voter, dtype=np.float64)
    
    # Check for NaN inputs
    if np.any(np.isnan(candidate)) or np.any(np.isnan(voter)):
        print(f"WARNING: NaN in input! candidate has {np.sum(np.isnan(candidate))} NaNs, "
              f"voter has {np.sum(np.isnan(voter))} NaNs")
        return 0.0
    
    # Elementwise products
    products = candidate * voter
    
    # Cube roots of products (handles negative numbers correctly)
    # np.cbrt is better than **(1/3) for negative numbers
    cube_roots = np.cbrt(products)
    
    # Check for NaN in intermediate results
    if np.any(np.isnan(cube_roots)):
        print(f"WARNING: NaN in cube_roots! products: {products}")
        # Replace NaNs with 0
        cube_roots = np.nan_to_num(cube_roots, nan=0.0)
    
    # Sum
    result = np.sum(cube_roots)
    
    # Final check
    if np.isnan(result):
        print(f"WARNING: NaN in final result! candidate: {candidate}, voter: {voter}")
        return 0.0
    
    return result

# %%
def voting_probability(voter_list, voter_index, candidate_list, candidate_index, 
                               p=0.5, default=False, epsilon=1e-10):
    """
    Computes the probability of a specific voter voting for a specific candidate.
    
    ROBUST VERSION: Handles numerical instability in softmax
    
    Parameters:
    -----------
    voter_list : array-like
        List of voter vectors
    voter_index : int
        Index of the specific voter
    candidate_list : array-like
        List of candidate vectors
    candidate_index : int
        Index of the specific candidate
    p : float, default=0.5
        Default probability constant (used when default=True)
    default : bool, default=False
        If True, returns constant probability p.
        If False, uses softmax based on utilities.
    epsilon : float, default=1e-10
        Small value for numerical stability
    
    Returns:
    --------
    float
        Probability of voter voting for candidate (guaranteed in [0, 1])
    """
    if default:
        return p
    
    # Get the specific voter
    voter = np.asarray(voter_list[voter_index], dtype=np.float64)
    
    # Calculate utilities for all candidates
    utilities = []
    for candidate in candidate_list:
        util = utility_function(candidate, voter)
        utilities.append(util)
    
    utilities = np.array(utilities, dtype=np.float64)
    
    # Check for NaN utilities
    if np.any(np.isnan(utilities)):
        print(f"WARNING: NaN utilities in voting_probability for voter {voter_index}")
        print(f"  utilities: {utilities}")
        # Replace NaNs with very negative number (low probability)
        utilities = np.nan_to_num(utilities, nan=-1000.0)
    
    # Apply softmax with numerical stability
    # Subtract max to prevent overflow
    max_util = np.max(utilities)
    
    # Check if max is too large/small
    if abs(max_util) > 700:  # exp(700) is near float overflow
        print(f"WARNING: Very large utility values (max={max_util}), rescaling")
        utilities = utilities / (abs(max_util) / 100)  # Rescale
        max_util = np.max(utilities)
    
    exp_utilities = np.exp(utilities - max_util)
    
    # Check for NaN or inf in exponentials
    if np.any(np.isnan(exp_utilities)) or np.any(np.isinf(exp_utilities)):
        print(f"WARNING: NaN/inf in exp(utilities)")
        print(f"  utilities: {utilities}")
        print(f"  exp_utilities: {exp_utilities}")
        exp_utilities = np.nan_to_num(exp_utilities, nan=epsilon, posinf=1e10, neginf=epsilon)
    
    sum_exp = np.sum(exp_utilities)
    
    # Prevent division by zero
    if sum_exp < epsilon:
        print(f"WARNING: Sum of exp_utilities too small: {sum_exp}")
        # Uniform probability as fallback
        return 1.0 / len(candidate_list)
    
    exp_utility_i = exp_utilities[candidate_index]
    probability = exp_utility_i / sum_exp
    
    # Final checks
    if np.isnan(probability):
        print(f"WARNING: NaN probability, returning uniform")
        return 1.0 / len(candidate_list)
    
    # Ensure probability is in valid range
    probability = np.clip(probability, 0.0, 1.0)
    
    return probability

# %%
def generate_voters(n, d, sparsity=0.5, seed=None):
    """
    Generate n voters with true utility vectors and sparse voting vectors.
    
    Parameters:
    -----------
    n : int
        Number of voters
    d : int
        Number of dimensions (policies)
    sparsity : float, default=0.5
        Fraction of dimensions to zero out in voting vector (0 = no sparsity, 1 = all zero)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    true_utility_vectors : ndarray
        Shape (n, d) - true utility vectors for each voter (all positive values)
    voting_vectors : ndarray
        Shape (n, d) - sparse voting vectors (some dimensions zeroed)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true utility vectors (only positive values)
    true_utility_vectors = np.random.uniform(0.1, 10.0, size=(n, d))
    
    # Create sparse voting vectors
    voting_vectors = true_utility_vectors.copy()
    
    # Zero out random dimensions for each voter
    for i in range(n):
        num_zeros = int(d * sparsity)
        zero_indices = np.random.choice(d, size=num_zeros, replace=False)
        voting_vectors[i, zero_indices] = 0
    
    return true_utility_vectors, voting_vectors


# %%
def generate_candidates(m, d, budget, seed=None):
    """
    Generate m candidates with policy vectors (budget allocation).
    
    Parameters:
    -----------
    m : int
        Number of candidates
    d : int
        Number of dimensions (policies)
    budget : float
        Total budget to allocate across dimensions
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    candidate_vectors : ndarray
        Shape (m, d) - policy vectors for each candidate
        Each dimension >= -1, and sum(vector) = budget (full budget is used)
    """
    if seed is not None:
        np.random.seed(seed)
    
    candidate_vectors = np.zeros((m, d))
    
    for i in range(m):
        # Start with all dimensions at -1 (sum = -d)
        # Need to add (budget + d) to get sum = budget
        # Allocate (budget + d) randomly across dimensions
        allocations = np.random.dirichlet(np.ones(d)) * (budget + d)
        
        # Add to -1 baseline: vector = -1 + allocations
        # This ensures: sum(vector) = -d + (budget + d) = budget
        # and each vector[i] >= -1 (since allocations[i] >= 0)
        candidate_vectors[i] = -1 + allocations
    
    # Verify budget constraint
    for i in range(m):
        assert np.all(candidate_vectors[i] >= -1), f"Candidate {i} violates lower bound"
        assert np.abs(np.sum(candidate_vectors[i]) - budget) < 1e-10, \
            f"Candidate {i} doesn't use full budget"
    
    return candidate_vectors


# %%
def run_approval_voting_simulation(n, m, d, budget, sparsity=0.5, seed=None):
    """
    Run a complete approval voting simulation using probabilistic voting.
    
    Parameters:
    -----------
    n : int
        Number of voters
    m : int
        Number of candidates
    d : int
        Number of dimensions (policies)
    budget : float
        Budget for each candidate
    sparsity : float, default=0.5
        Sparsity of voting vectors (fraction of dimensions zeroed)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'vote_counts': array of vote counts for each candidate
        - 'true_utility_vectors': voter true utility vectors
        - 'voting_vectors': voter voting vectors
        - 'candidate_vectors': candidate policy vectors
        - 'approval_matrix': (n, m) boolean matrix of approvals
    """
    # Generate voters
    true_utility_vectors, voting_vectors = generate_voters(n, d, sparsity, seed)
    
    # Generate candidates
    candidate_vectors = generate_candidates(m, d, budget, seed)
    
    # Convert to lists for voting_probability function
    voting_vectors_list = [voting_vectors[i] for i in range(n)]
    candidate_vectors_list = [candidate_vectors[i] for i in range(m)]
    
    # Determine approvals using probabilistic voting
    approval_matrix = np.zeros((n, m), dtype=bool)
    
    for voter_idx in range(n):
        for candidate_idx in range(m):
            # Get voting probability using softmax
            prob = voting_probability(
                voting_vectors_list, 
                voter_index=voter_idx,
                candidate_list=candidate_vectors_list,
                candidate_index=candidate_idx,
                default=False
            )
            
            # Toss weighted coin: approve with probability prob
            approval = np.random.rand() < prob
            approval_matrix[voter_idx, candidate_idx] = approval
    
    # Count votes for each candidate
    vote_counts = np.sum(approval_matrix, axis=0)
    
    results = {
        'vote_counts': vote_counts,
        'true_utility_vectors': true_utility_vectors,
        'voting_vectors': voting_vectors,
        'candidate_vectors': candidate_vectors,
        'approval_matrix': approval_matrix
    }
    
    return results


# %%
def select_winner_and_compute_global_utility(vote_counts, candidate_vectors, true_utility_vectors, k, seed=None):
    """
    Select winner from top k candidates and compute global utility.
    
    Parameters:
    -----------
    vote_counts : array-like
        Vote counts for each candidate
    candidate_vectors : array-like
        Policy vectors for each candidate
    true_utility_vectors : array-like
        True utility vectors for each voter
    k : int
        Number of top candidates to consider
    seed : int, optional
        Random seed for winner selection
    
    Returns:
    --------
    winner_idx : int
        Index of the winning candidate
    global_utility : float
        Sum of utilities across all voters for the winner
    top_k_indices : array
        Indices of the top k candidates
    """
    if seed is not None:
        np.random.seed(seed)
    
    vote_counts = np.asarray(vote_counts)
    
    # Find top k candidates (handle ties by taking first k)
    # Sort indices by vote count in descending order
    sorted_indices = np.argsort(vote_counts)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Randomly select winner from top k (uniform probability 1/k)
    winner_idx = np.random.choice(top_k_indices)
    
    # Calculate global utility using true utility vectors
    winner_vector = candidate_vectors[winner_idx]
    global_utility = 0.0
    
    for voter_idx in range(len(true_utility_vectors)):
        true_utility_vector = true_utility_vectors[voter_idx]
        voter_utility = utility_function(winner_vector, true_utility_vector)
        global_utility += voter_utility
    
    return winner_idx, global_utility, top_k_indices


# %%
def estimate_expected_global_utility_by_k(n, m, d, budget, sparsity=0.5, n_simulations=100, seed=None):
    """
    Estimate expected global utility for each k (1 to m) using Monte Carlo simulation.
    
    Parameters:
    -----------
    n : int
        Number of voters
    m : int
        Number of candidates
    d : int
        Number of dimensions
    budget : float
        Budget for each candidate
    sparsity : float, default=0.5
        Sparsity of voting vectors
    n_simulations : int, default=100
        Number of Monte Carlo simulations to run
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    expected_utilities : dict
        Dictionary mapping k -> expected global utility
    std_utilities : dict
        Dictionary mapping k -> standard deviation of global utility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Store utilities for each k across all simulations
    utilities_by_k = {k: [] for k in range(1, m + 1)}
    
    for sim in range(n_simulations):
        # Run simulation with different seed for each run
        results = run_approval_voting_simulation(
            n=n, m=m, d=d, budget=budget, sparsity=sparsity, seed=None
        )
        
        # For each possible k, compute expected global utility
        vote_counts = results['vote_counts']
        candidate_vectors = results['candidate_vectors']
        true_utility_vectors = results['true_utility_vectors']
        
        # Sort candidates by vote count
        sorted_indices = np.argsort(vote_counts)[::-1]
        
        # For each k, compute expected utility
        for k in range(1, m + 1):
            top_k_indices = sorted_indices[:k]
            
            # Expected utility = average utility of top k candidates
            # (since winner is randomly selected with probability 1/k)
            expected_utility = 0.0
            for candidate_idx in top_k_indices:
                candidate_vector = candidate_vectors[candidate_idx]
                global_utility = 0.0
                
                for voter_idx in range(len(true_utility_vectors)):
                    true_utility_vector = true_utility_vectors[voter_idx]
                    voter_utility = utility_function(candidate_vector, true_utility_vector)
                    global_utility += voter_utility
                
                expected_utility += global_utility / k  # Each has probability 1/k
            
            utilities_by_k[k].append(expected_utility)
    
    # Compute means and standard deviations
    expected_utilities = {k: np.mean(utilities_by_k[k]) for k in range(1, m + 1)}
    std_utilities = {k: np.std(utilities_by_k[k]) for k in range(1, m + 1)}
    
    return expected_utilities, std_utilities


# %%
def find_optimal_k(n, m, d, budget, sparsity=0.5, n_simulations=100, seed=None):
    """
    Find the optimal k that maximizes expected global utility.
    
    Parameters:
    -----------
    n : int
        Number of voters
    m : int
        Number of candidates
    d : int
        Number of dimensions
    budget : float
        Budget for each candidate
    sparsity : float, default=0.5
        Sparsity of voting vectors
    n_simulations : int, default=100
        Number of Monte Carlo simulations to run
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    optimal_k : int
        The value of k that maximizes expected global utility
    expected_utilities : dict
        Dictionary mapping k -> expected global utility
    std_utilities : dict
        Dictionary mapping k -> standard deviation of global utility
    """
    expected_utilities, std_utilities = estimate_expected_global_utility_by_k(
        n, m, d, budget, sparsity, n_simulations, seed
    )
    
    # Find k with maximum expected utility
    optimal_k = max(expected_utilities, key=expected_utilities.get)
    
    return optimal_k, expected_utilities, std_utilities


# %%
def utility_gradient(candidate, voter, epsilon=1e-8):
    """
    Compute the gradient of utility with respect to candidate position.
    
    ROBUST VERSION: Prevents division by zero and NaN propagation
    
    The derivative of cbrt(x) = (1/3) * x^(-2/3) = (1/3) / x^(2/3)
    For products close to zero, this explodes, so we clip it.
    
    Parameters:
    -----------
    candidate : array-like
        Candidate policy vector
    voter : array-like
        Voter vector
    epsilon : float, default=1e-8
        Threshold for considering products as zero
    
    Returns:
    --------
    gradient : ndarray
        Gradient of utility w.r.t. candidate position (guaranteed not NaN)
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    voter = np.asarray(voter, dtype=np.float64)
    
    # Check for NaN inputs
    if np.any(np.isnan(candidate)) or np.any(np.isnan(voter)):
        print(f"WARNING: NaN in gradient input!")
        return np.zeros_like(candidate)
    
    products = candidate * voter
    gradient = np.zeros_like(candidate, dtype=np.float64)
    
    for k in range(len(candidate)):
        # Skip if voter component is zero (gradient is zero)
        if abs(voter[k]) < epsilon:
            gradient[k] = 0.0
            continue
        
        # Handle near-zero products carefully
        if abs(products[k]) < epsilon:
            # Near zero, gradient is very large but we clip it
            # Sign depends on sign of voter[k]
            gradient[k] = np.sign(voter[k]) * 1000.0  # Clipped large value
            continue
        
        # Normal case: d/dc[k] cbrt(c[k] * v[k]) = (1/3) * v[k] / (c[k] * v[k])^(2/3)
        # Use sign-preserving power for negative products
        sign = np.sign(products[k])
        abs_product = abs(products[k])
        
        # Compute x^(-2/3) = 1 / x^(2/3) safely
        power_term = abs_product ** (2.0/3.0)
        
        if power_term < epsilon:
            # Very small denominator, clip gradient
            gradient[k] = np.sign(voter[k]) * 1000.0
        else:
            gradient[k] = (1.0/3.0) * voter[k] / (sign * power_term)
    
    # Final NaN check
    if np.any(np.isnan(gradient)):
        print(f"WARNING: NaN in gradient output! Replacing with zeros.")
        print(f"  candidate: {candidate}")
        print(f"  voter: {voter}")
        print(f"  products: {products}")
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    # Clip extremely large gradients
    gradient = np.clip(gradient, -1000.0, 1000.0)
    
    return gradient


# %%
def top_k_win_probability_gradient(candidate_idx, candidate_vectors, voting_vectors, k, n_samples=100):
    """
    Compute the gradient of (P(in top k) * 1/k) for a candidate.
    
    This is the expected value of winning: if candidate is in top k, they have 1/k chance.
    
    Uses Monte Carlo sampling to estimate the probability and its gradient.
    
    Parameters:
    -----------
    candidate_idx : int
        Index of the candidate
    candidate_vectors : array-like, shape (m, d)
        All candidate policy vectors
    voting_vectors : array-like, shape (n, d)
        Voting vectors for all voters
    k : int
        Number of top candidates to consider
    n_samples : int, default=100
        Number of Monte Carlo samples for gradient estimation
    
    Returns:
    --------
    gradient : ndarray
        Gradient of (P(in top k) * 1/k) w.r.t. candidate position
    """
    candidate_vectors = np.asarray(candidate_vectors)
    voting_vectors = np.asarray(voting_vectors)
    candidate = candidate_vectors[candidate_idx]
    n_voters = len(voting_vectors)
    m_candidates = len(candidate_vectors)
    
    # Convert to lists for voting_probability
    voting_vectors_list = [voting_vectors[i] for i in range(n_voters)]
    candidate_vectors_list = [candidate_vectors[i] for i in range(m_candidates)]
    
    # Compute approval probabilities for all candidate-voter pairs
    approval_probs = np.zeros((n_voters, m_candidates))
    for voter_idx in range(n_voters):
        for cand_idx in range(m_candidates):
            approval_probs[voter_idx, cand_idx] = voting_probability(
                voting_vectors_list,
                voter_index=voter_idx,
                candidate_list=candidate_vectors_list,
                candidate_index=cand_idx,
                default=False
            )
    
    # Use finite differences with small perturbations
    epsilon = 1e-5
    gradient = np.zeros_like(candidate)
    
    # Estimate gradient using finite differences
    for dim in range(len(candidate)):
        # Perturb candidate position
        candidate_perturbed = candidate.copy()
        candidate_perturbed[dim] += epsilon
        
        # Create perturbed candidate vectors
        candidate_vectors_perturbed = candidate_vectors.copy()
        candidate_vectors_perturbed[candidate_idx] = candidate_perturbed
        candidate_vectors_list_perturbed = [candidate_vectors_perturbed[i] for i in range(m_candidates)]
        
        # Recompute approval probabilities with perturbation
        approval_probs_perturbed = np.zeros((n_voters, m_candidates))
        for voter_idx in range(n_voters):
            for cand_idx in range(m_candidates):
                approval_probs_perturbed[voter_idx, cand_idx] = voting_probability(
                    voting_vectors_list,
                    voter_index=voter_idx,
                    candidate_list=candidate_vectors_list_perturbed,
                    candidate_index=cand_idx,
                    default=False
                )
        
        # Monte Carlo estimate of P(in top k) for original and perturbed
        prob_original = estimate_top_k_probability(approval_probs, candidate_idx, k, n_samples)
        prob_perturbed = estimate_top_k_probability(approval_probs_perturbed, candidate_idx, k, n_samples)
        
        # Finite difference gradient
        # The objective is P(in top k) * (1/k) = expected probability of winning
        gradient[dim] = (prob_perturbed - prob_original) / epsilon * (1.0 / k)
    
    return gradient


def estimate_top_k_probability(approval_probs, candidate_idx, k, n_samples=100):
    """
    Estimate P(candidate is in top k) using Monte Carlo sampling.
    
    Parameters:
    -----------
    approval_probs : ndarray, shape (n_voters, m_candidates)
        Probability each voter approves each candidate
    candidate_idx : int
        Index of candidate of interest
    k : int
        Number of top candidates
    n_samples : int, default=100
        Number of Monte Carlo samples
    
    Returns:
    --------
    probability : float
        Estimated probability candidate is in top k
    """
    n_voters, m_candidates = approval_probs.shape
    in_top_k_count = 0
    
    for _ in range(n_samples):
        # Sample vote counts: for each voter-candidate pair, sample approval
        vote_counts = np.zeros(m_candidates)
        for voter_idx in range(n_voters):
            for cand_idx in range(m_candidates):
                if np.random.rand() < approval_probs[voter_idx, cand_idx]:
                    vote_counts[cand_idx] += 1
        
        # Check if candidate is in top k
        sorted_indices = np.argsort(vote_counts)[::-1]
        top_k_indices = sorted_indices[:k]
        if candidate_idx in top_k_indices:
            in_top_k_count += 1
    
    return in_top_k_count / n_samples

# %%
def expected_approval_gradient(candidate_idx, candidate_vectors, voting_vectors):
    """
    Compute the gradient of expected approval votes for a candidate.
    
    The expected approval votes = sum over voters of P(voter approves candidate)
    where P(voter approves candidate) is given by voting_probability (softmax).
    
    Why we need utility_gradient:
    - P depends on utility: P = softmax(utilities) = exp(u_i) / sum(exp(u_j))
    - Utility depends on candidate position: u_i = utility_function(candidate, voter)
    - To get dP/dc, we use the chain rule: dP/dc = (dP/du) * (du/dc)
      where:
      - dP/du = p_i * (1 - p_i) (derivative of softmax)
      - du/dc = utility_gradient (derivative of utility function)
    
    Parameters:
    -----------
    candidate_idx : int
        Index of the candidate
    candidate_vectors : array-like, shape (m, d)
        All candidate policy vectors
    voting_vectors : array-like, shape (n, d)
        Voting vectors for all voters
    
    Returns:
    --------
    gradient : ndarray
        Gradient of expected approval votes w.r.t. candidate position
    """
    candidate_vectors = np.asarray(candidate_vectors)
    voting_vectors = np.asarray(voting_vectors)
    candidate = candidate_vectors[candidate_idx]
    
    n_voters = len(voting_vectors)
    gradient = np.zeros_like(candidate)
    
    # Convert to lists for voting_probability function
    voting_vectors_list = [voting_vectors[i] for i in range(n_voters)]
    candidate_vectors_list = [candidate_vectors[i] for i in range(len(candidate_vectors))]
    
    for voter_idx in range(n_voters):
        voter = voting_vectors[voter_idx]
        
        # Get the voting probability using the same function as in simulation
        # This is P(voter approves candidate) = softmax probability
        p_i = voting_probability(
            voting_vectors_list,
            voter_index=voter_idx,
            candidate_list=candidate_vectors_list,
            candidate_index=candidate_idx,
            default=False
        )
        
        # CHAIN RULE: To compute dP/dc, we need:
        # 1. dP/du = how probability changes with utility (softmax derivative)
        # 2. du/dc = how utility changes with candidate position (utility_gradient)
        # Then: dP/dc = (dP/du) * (du/dc) = p_i * (1 - p_i) * du_i/dc
        
        # Step 1: dP/du for softmax is p_i * (1 - p_i)
        # (This is the derivative of softmax probability w.r.t. its own utility)
        
        # Step 2: du/dc = how utility changes when we change candidate position
        # This is what utility_gradient computes
        du_i_dc = utility_gradient(candidate, voter)
        
        # Apply chain rule: dP/dc = (dP/du) * (du/dc)
        dp_i_dc = p_i * (1 - p_i) * du_i_dc
        
        # Gradient of expected approval: sum over voters of dp_i/dc
        gradient += dp_i_dc
    
    return gradient


# %%
def project_to_constraints(candidate, budget, lower_bound=-1.0, epsilon=1e-6):
    """
    Project a candidate vector to satisfy budget and lower bound constraints.
    
    ROBUST VERSION: Ensures numerical stability and valid outputs
    
    Parameters:
    -----------
    candidate : array-like
        Candidate policy vector
    budget : float
        Budget constraint (sum of vector must equal this)
    lower_bound : float, default=-1.0
        Lower bound for each dimension
    epsilon : float, default=1e-6
        Tolerance for constraint satisfaction
    
    Returns:
    --------
    projected : ndarray
        Projected candidate vector satisfying constraints (guaranteed not NaN)
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    
    # Check for NaN inputs
    if np.any(np.isnan(candidate)):
        print(f"WARNING: NaN in project_to_constraints input, using uniform allocation")
        d = len(candidate)
        return np.full(d, lower_bound + (budget - lower_bound * d) / d)
    
    d = len(candidate)
    
    # First, enforce lower bound
    candidate = np.maximum(candidate, lower_bound)
    
    # Then, adjust to satisfy budget constraint
    current_sum = np.sum(candidate)
    
    if np.isnan(current_sum):
        print(f"WARNING: NaN in current_sum during projection")
        return np.full(d, lower_bound + (budget - lower_bound * d) / d)
    
    if abs(current_sum - budget) > epsilon:
        # Amount to adjust
        diff = budget - current_sum
        
        # Find dimensions that can be adjusted (those above lower bound)
        adjustable = candidate > (lower_bound + epsilon)
        n_adjustable = np.sum(adjustable)
        
        if n_adjustable > 0:
            # Distribute the difference proportionally
            current_values = candidate[adjustable]
            if np.sum(current_values) > epsilon:
                # Proportional adjustment
                adjustment_weights = current_values / np.sum(current_values)
                candidate[adjustable] += diff * adjustment_weights
            else:
                # Uniform adjustment
                candidate[adjustable] += diff / n_adjustable
            
            # Re-enforce lower bound
            candidate = np.maximum(candidate, lower_bound)
        else:
            # All at lower bound, distribute remaining budget uniformly
            remaining = budget - d * lower_bound
            if remaining > 0:
                candidate = np.full(d, lower_bound + remaining / d)
            else:
                print(f"WARNING: Budget {budget} cannot satisfy {d} dims at lower bound {lower_bound}")
                candidate = np.full(d, lower_bound)
    
    # Final verification
    final_sum = np.sum(candidate)
    if abs(final_sum - budget) > epsilon:
        print(f"WARNING: Projection failed to satisfy budget: {final_sum} vs {budget}")
        # Force correction
        candidate = candidate * (budget / final_sum) if final_sum > epsilon else np.full(d, budget / d)
    
    # Check for NaN in output
    if np.any(np.isnan(candidate)):
        print(f"WARNING: NaN in projection output, using uniform allocation")
        return np.full(d, lower_bound + (budget - lower_bound * d) / d)
    
    return candidate


# %%
def optimize_candidates(candidate_vectors, voting_vectors, budget, k, learning_rate=0.1, 
                        convergence_threshold=1e-2, max_iterations=1000, seed=None, verbose=True):
    """
    Optimize candidate positions using gradient descent to maximize P(in top k) * (1/k).
    
    Parameters:
    -----------
    candidate_vectors : array-like, shape (m, d)
        Initial candidate policy vectors
    voting_vectors : array-like, shape (n, d)
        Voting vectors for all voters
    budget : float
        Budget constraint for each candidate
    k : int
        Number of top candidates (candidates optimize for P(in top k) * 1/k)
    learning_rate : float, default=0.1
        Learning rate for gradient descent
    convergence_threshold : float, default=1e-2
        Convergence threshold (gradient magnitude)
    max_iterations : int, default=1000
        Maximum number of iterations per candidate
    seed : int, optional
        Random seed for candidate ordering
    verbose : bool, default=True
        Whether to print progress
    
    Returns:
    --------
    optimized_vectors : ndarray
        Optimized candidate policy vectors
    history : list
        List of candidate vectors at each iteration
    """
    if seed is not None:
        np.random.seed(seed)
    
    candidate_vectors = np.asarray(candidate_vectors).copy()
    voting_vectors = np.asarray(voting_vectors)
    m_candidates, d_dimensions = candidate_vectors.shape
    
    # Random ordering of candidates
    candidate_order = np.random.permutation(m_candidates)
    
    if verbose:
        print(f"Optimizing {m_candidates} candidates for top-{k} selection...")
        print(f"Order: {candidate_order + 1}")  # 1-indexed for display
    
    history = [candidate_vectors.copy()]
    converged = np.zeros(m_candidates, dtype=bool)
    
    iteration = 0
    while not np.all(converged) and iteration < max_iterations:
        iteration += 1
        any_update = False
        
        for candidate_idx in candidate_order:
            if converged[candidate_idx]:
                continue
            
            # Compute gradient using top-k win probability
            gradient = top_k_win_probability_gradient(
                candidate_idx, candidate_vectors, voting_vectors, k
            )
            
            # Check convergence
            gradient_magnitude = np.linalg.norm(gradient)
            if gradient_magnitude < convergence_threshold:
                converged[candidate_idx] = True
                continue
            
            # Update candidate position
            candidate_vectors[candidate_idx] += learning_rate * gradient
            
            # Project to constraints
            candidate_vectors[candidate_idx] = project_to_constraints(
                candidate_vectors[candidate_idx], budget
            )
            
            any_update = True
        
        if any_update:
            history.append(candidate_vectors.copy())
        
        if verbose and iteration % 100 == 0:
            n_converged = np.sum(converged)
            print(f"Iteration {iteration}: {n_converged}/{m_candidates} candidates converged")
    
    if verbose:
        if np.all(converged):
            print(f"All candidates converged after {iteration} iterations")
        else:
            print(f"Stopped after {max_iterations} iterations ({np.sum(converged)}/{m_candidates} converged)")
    
    return candidate_vectors, history

# %%
def run_simulation_with_optimization_all_k(voting_vectors, true_utility_vectors, n_candidates, budget,
                                           n_simulations=1000, learning_rate=0.1, convergence_threshold=1e-2,
                                           max_iterations=1000, optimization_seed=None, simulation_seed=None, 
                                           verbose=True):
    """
    Run simulation with optimization for ALL values of k, then report expected global utility for each k.
    
    For each k from 1 to m:
    1. Optimize candidates to maximize P(in top k) * (1/k)
    2. Run simulations to compute expected global utility when winner is randomly selected from top k
    
    FIXED: Now randomly selects ONE winner from top-k instead of averaging over all top-k candidates.
    This gives non-zero variance even when the same candidates are always in top-k.
    
    Parameters:
    -----------
    voting_vectors : array-like, shape (n_voters, d_dimensions)
        Voting vectors for each voter
    true_utility_vectors : array-like, shape (n_voters, d_dimensions)
        True utility vectors for each voter
    n_candidates : int
        Number of candidates
    budget : float
        Budget constraint for each candidate
    n_simulations : int, default=1000
        Number of simulations for expected utility calculation
    learning_rate : float, default=0.1
        Learning rate for gradient descent
    convergence_threshold : float, default=1e-2
        Convergence threshold for optimization
    max_iterations : int, default=1000
        Maximum iterations for optimization
    optimization_seed : int, optional
        Random seed for candidate initialization and ordering
    simulation_seed : int, optional
        Random seed for simulation
    verbose : bool, default=True
        Whether to print progress
    
    Returns:
    --------
    results_by_k : dict
        Dictionary mapping k -> results for that k value
        Each results dict contains:
        - 'optimized_candidates': optimized candidate vectors
        - 'expected_utility': expected global utility (mean across simulations)
        - 'std_utility': standard deviation (now non-zero!)
        - 'all_utilities': all utility values from simulations
    """
    voting_vectors = np.asarray(voting_vectors)
    true_utility_vectors = np.asarray(true_utility_vectors)
    n_voters, d_dimensions = voting_vectors.shape
    
    results_by_k = {}
    
    if verbose:
        print("=" * 60)
        print("OPTIMIZATION FOR ALL K VALUES")
        print("=" * 60)
        print(f"Voters: {n_voters}, Candidates: {n_candidates}, Dimensions: {d_dimensions}")
        print(f"Budget: {budget}")
        print()
    
    for k in range(1, n_candidates + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"K = {k}")
            print(f"{'=' * 60}\n")
        
        # Initialize candidates randomly (same seed for each k for fair comparison)
        if optimization_seed is not None:
            np.random.seed(optimization_seed)
        
        initial_candidates = generate_candidates(n_candidates, d_dimensions, budget, seed=optimization_seed)
        
        # Optimize candidates for this k
        optimized_candidates, _ = optimize_candidates(
            initial_candidates,
            voting_vectors,
            budget,
            k=k,
            learning_rate=learning_rate,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
            seed=optimization_seed,
            verbose=verbose
        )
        
        # Run simulations with optimized candidates
        if verbose:
            print(f"\nRunning {n_simulations} simulations for k={k}...")
        
        # Set simulation seed if provided
        if simulation_seed is not None:
            np.random.seed(simulation_seed)
        
        utilities = []
        for sim in range(n_simulations):
            # Run probabilistic voting
            approval_matrix = np.zeros((n_voters, n_candidates), dtype=bool)
            voting_vectors_list = [voting_vectors[i] for i in range(n_voters)]
            candidate_vectors_list = [optimized_candidates[i] for i in range(n_candidates)]
            
            for voter_idx in range(n_voters):
                for candidate_idx in range(n_candidates):
                    prob = voting_probability(
                        voting_vectors_list, 
                        voter_index=voter_idx,
                        candidate_list=candidate_vectors_list,
                        candidate_index=candidate_idx,
                        default=False
                    )
                    approval = np.random.rand() < prob
                    approval_matrix[voter_idx, candidate_idx] = approval
            
            vote_counts = np.sum(approval_matrix, axis=0)
            
            # Get top k candidates
            sorted_indices = np.argsort(vote_counts)[::-1]
            top_k_indices = sorted_indices[:k]
            
            # ============================================================
            # FIXED: Randomly select ONE winner from top-k
            # (Instead of averaging over all top-k candidates)
            # ============================================================
            winner_idx = np.random.choice(top_k_indices)
            
            # Compute actual global utility for THIS winner
            winner_vector = optimized_candidates[winner_idx]
            global_utility = 0.0
            
            for voter_idx in range(n_voters):
                true_utility_vector = true_utility_vectors[voter_idx]
                voter_utility = utility_function(winner_vector, true_utility_vector)
                global_utility += voter_utility
            
            # Store the actual utility (not expected utility)
            utilities.append(global_utility)
        
        # Store results
        results_by_k[k] = {
            'optimized_candidates': optimized_candidates,
            'expected_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
            'all_utilities': utilities
        }
        
        if verbose:
            print(f"Expected global utility for k={k}: {np.mean(utilities):.4f} ± {np.std(utilities):.4f}")
    
    # Print summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("SUMMARY: Expected Global Utility by K")
        print(f"{'=' * 60}\n")
        
        optimal_k = max(results_by_k.keys(), key=lambda k: results_by_k[k]['expected_utility'])
        
        for k in range(1, n_candidates + 1):
            marker = " <-- OPTIMAL" if k == optimal_k else ""
            print(f"k={k}: {results_by_k[k]['expected_utility']:.4f} ± {results_by_k[k]['std_utility']:.4f}{marker}")
    
    return results_by_k


# %%
# Example simulation
n_voters = 20
m_candidates = 5
d_dimensions = 10
budget = 20.0
sparsity = 0.4  # 40% of dimensions zeroed out in voting vectors
k = 3  # Top k candidates to consider for winner selection

results = run_approval_voting_simulation(
    n=n_voters,
    m=m_candidates,
    d=d_dimensions,
    budget=budget,
    sparsity=sparsity,
    seed=42
)

print("Approval Voting Simulation Results")
print("=" * 50)
print(f"Voters: {n_voters}, Candidates: {m_candidates}, Dimensions: {d_dimensions}")
print(f"Budget per candidate: {budget}, Voting vector sparsity: {sparsity}")
print("\nVote counts for each candidate:")
for i, votes in enumerate(results['vote_counts']):
    print(f"  Candidate {i+1}: {votes} votes ({votes/n_voters*100:.1f}% approval rate)")

print(f"\nTotal votes cast: {np.sum(results['vote_counts'])}")
print(f"Average approvals per voter: {np.sum(results['approval_matrix']) / n_voters:.2f}")

# Select winner from top k and compute global utility
winner_idx, global_utility, top_k_indices = select_winner_and_compute_global_utility(
    results['vote_counts'],
    results['candidate_vectors'],
    results['true_utility_vectors'],
    k=k,
    seed=42
)

print(f"\nTop {k} candidates (by vote count): {[i+1 for i in top_k_indices]}")
print(f"Winner (randomly selected from top {k}): Candidate {winner_idx+1}")
print(f"Global utility (using true utility vectors): {global_utility:.4f}")
print(f"Average utility per voter: {global_utility/n_voters:.4f}")


# %%
def generate_clustered_voters(n, d, n_clusters=3, cluster_size_variance=0.2, sparsity=0.3, seed=None):
    """
    Generate voters with voting vectors clustered into groups.
    
    Parameters:
    -----------
    n : int
        Number of voters
    d : int
        Number of dimensions
    n_clusters : int, default=3
        Number of clusters
    cluster_size_variance : float, default=0.2
        Variance in cluster sizes (0 = equal sizes)
    sparsity : float, default=0.3
        Sparsity of voting vectors (fraction of dimensions zeroed)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    true_utility_vectors : ndarray
        True utility vectors (all positive, random)
    voting_vectors : ndarray
        Clustered voting vectors
    cluster_labels : ndarray
        Cluster assignment for each voter
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centers (each cluster focuses on different dimensions)
    cluster_centers = np.zeros((n_clusters, d))
    dimensions_per_cluster = d // n_clusters
    
    for cluster_idx in range(n_clusters):
        # Each cluster focuses on a different set of dimensions
        start_dim = cluster_idx * dimensions_per_cluster
        end_dim = min((cluster_idx + 1) * dimensions_per_cluster, d)
        # Set some dimensions to positive values
        cluster_centers[cluster_idx, start_dim:end_dim] = np.random.uniform(0.5, 2.0, size=end_dim - start_dim)
    
    # Assign voters to clusters
    cluster_sizes = np.random.multinomial(n, [1/n_clusters] * n_clusters)
    cluster_sizes = cluster_sizes + 1  # Ensure at least 1 voter per cluster
    cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n).astype(int)
    # Adjust to ensure sum equals n
    cluster_sizes[-1] += n - cluster_sizes.sum()
    
    cluster_labels = []
    for cluster_idx, size in enumerate(cluster_sizes):
        cluster_labels.extend([cluster_idx] * size)
    cluster_labels = np.array(cluster_labels)
    np.random.shuffle(cluster_labels)
    
    # Generate voting vectors around cluster centers
    voting_vectors = np.zeros((n, d))
    for i in range(n):
        cluster_idx = cluster_labels[i]
        center = cluster_centers[cluster_idx]
        # Add noise around cluster center
        noise = np.random.normal(0, 0.3, size=d)
        voting_vectors[i] = center + noise
        # Ensure non-negative (since voting vectors should be non-negative)
        voting_vectors[i] = np.maximum(voting_vectors[i], 0)
        
        # Apply sparsity: zero out some dimensions
        num_zeros = int(d * sparsity)
        zero_indices = np.random.choice(d, size=num_zeros, replace=False)
        voting_vectors[i, zero_indices] = 0
    
    # Generate true utility vectors clustered around similar patterns
    # True utilities care about all dimensions, but still clustered
    # Each cluster has a base pattern, but all dimensions have some value
    true_utility_centers = np.zeros((n_clusters, d))
    for cluster_idx in range(n_clusters):
        # Base pattern similar to voting cluster, but all dimensions have values
        # The cluster's focus dimensions get higher values
        start_dim = cluster_idx * dimensions_per_cluster
        end_dim = min((cluster_idx + 1) * dimensions_per_cluster, d)
        
        # Focus dimensions get higher values
        true_utility_centers[cluster_idx, start_dim:end_dim] = np.random.uniform(3.0, 8.0, size=end_dim - start_dim)
        # Other dimensions get lower but still positive values
        other_dims = np.ones(d, dtype=bool)
        other_dims[start_dim:end_dim] = False
        true_utility_centers[cluster_idx, other_dims] = np.random.uniform(1.0, 4.0, size=np.sum(other_dims))
    
    # Generate true utility vectors around cluster centers
    true_utility_vectors = np.zeros((n, d))
    for i in range(n):
        cluster_idx = cluster_labels[i]
        center = true_utility_centers[cluster_idx]
        # Add noise around cluster center
        noise = np.random.normal(0, 0.5, size=d)
        true_utility_vectors[i] = center + noise
        # Ensure all positive (true utilities should be positive)
        true_utility_vectors[i] = np.maximum(true_utility_vectors[i], 0.1)
    
    return true_utility_vectors, voting_vectors, cluster_labels


# # %%
# # ============================================================
# # EXAMPLE USAGE
# # ============================================================

# # Example 1: Simple clustered voters
# print("EXAMPLE 1: Clustered Voters (3 groups)")
# print("=" * 60)

# n_voters = 30
# n_candidates = 4
# d_dimensions = 6
# budget = 10.0
# n_clusters = 3
# sparsity = 0.3

# # Generate clustered voters
# true_utility_vectors, voting_vectors, cluster_labels = generate_clustered_voters(
#     n=n_voters,
#     d=d_dimensions,
#     n_clusters=n_clusters,
#     cluster_size_variance=0.2,
#     sparsity=sparsity,
#     seed=123
# )

# print(f"Generated {n_voters} voters in {n_clusters} clusters")
# print(f"Cluster sizes: {[np.sum(cluster_labels == i) for i in range(n_clusters)]}")
# print()

# # Run optimization for all k values
# results_by_k = run_simulation_with_optimization_all_k(
#     voting_vectors=voting_vectors,
#     true_utility_vectors=true_utility_vectors,
#     n_candidates=n_candidates,
#     budget=budget,
#     n_simulations=200,  # Fewer simulations for faster example
#     learning_rate=0.05,
#     convergence_threshold=1e-2,
#     max_iterations=200,  # Fewer iterations for faster example
#     optimization_seed=456,
#     simulation_seed=789,
#     verbose=True
# )

# # Access results for specific k
# print("\n" + "=" * 60)
# print("DETAILED RESULTS FOR K=2")
# print("=" * 60)
# k = 2
# print(f"Optimized candidates for k={k}:")
# for i, candidate in enumerate(results_by_k[k]['optimized_candidates']):
#     print(f"  Candidate {i+1}: {candidate}")
# print(f"\nExpected utility: {results_by_k[k]['expected_utility']:.4f}")
# print(f"Std deviation: {results_by_k[k]['std_utility']:.4f}")


# # ============================================================
# # Example 2: Random voters (non-clustered)
# print("\n\n" + "=" * 60)
# print("EXAMPLE 2: Random Voters (non-clustered)")
# print("=" * 60)

# n_voters = 25
# n_candidates = 3
# d_dimensions = 5
# budget = 8.0
# sparsity = 0.4

# # Generate random voters
# true_utility_vectors, voting_vectors = generate_voters(
#     n=n_voters,
#     d=d_dimensions,
#     sparsity=sparsity,
#     seed=999
# )

# print(f"Generated {n_voters} random voters")
# print()

# # Run optimization for all k values
# results_by_k_random = run_simulation_with_optimization_all_k(
#     voting_vectors=voting_vectors,
#     true_utility_vectors=true_utility_vectors,
#     n_candidates=n_candidates,
#     budget=budget,
#     n_simulations=150,
#     learning_rate=0.05,
#     convergence_threshold=1e-2,
#     max_iterations=150,
#     optimization_seed=111,
#     simulation_seed=222,
#     verbose=True
# )

# # Compare k=1 vs k=m (all candidates)
# print("\n" + "=" * 60)
# print("COMPARISON: k=1 vs k=m")
# print("=" * 60)
# print(f"k=1 (winner takes all): {results_by_k_random[1]['expected_utility']:.4f}")
# print(f"k={n_candidates} (all candidates): {results_by_k_random[n_candidates]['expected_utility']:.4f}")
# improvement = results_by_k_random[1]['expected_utility'] - results_by_k_random[n_candidates]['expected_utility']
# print(f"Difference: {improvement:.4f} ({improvement/results_by_k_random[n_candidates]['expected_utility']*100:.1f}%)")

# # %%
# # ============================================================
# # DIAGNOSTIC FUNCTION
# # ============================================================

# def diagnose_nan_issues(candidate_vectors, voting_vectors, true_utility_vectors):
#     """
#     Run diagnostics to find sources of NaN values.
    
#     Parameters:
#     -----------
#     candidate_vectors : ndarray
#         Candidate policy vectors
#     voting_vectors : ndarray
#         Voting vectors
#     true_utility_vectors : ndarray
#         True utility vectors
    
#     Returns:
#     --------
#     dict : Diagnostic results
#     """
#     print("=" * 60)
#     print("NaN DIAGNOSTIC REPORT")
#     print("=" * 60)
    
#     issues = {
#         'nan_in_candidates': False,
#         'nan_in_voting': False,
#         'nan_in_true_utility': False,
#         'nan_in_utilities': False,
#         'nan_in_gradients': False,
#         'extreme_values': False
#     }
    
#     # Check inputs
#     if np.any(np.isnan(candidate_vectors)):
#         issues['nan_in_candidates'] = True
#         print(f"❌ Found {np.sum(np.isnan(candidate_vectors))} NaN values in candidate_vectors")
#     else:
#         print(f"✓ No NaN in candidate_vectors")
    
#     if np.any(np.isnan(voting_vectors)):
#         issues['nan_in_voting'] = True
#         print(f"❌ Found {np.sum(np.isnan(voting_vectors))} NaN values in voting_vectors")
#     else:
#         print(f"✓ No NaN in voting_vectors")
    
#     if np.any(np.isnan(true_utility_vectors)):
#         issues['nan_in_true_utility'] = True
#         print(f"❌ Found {np.sum(np.isnan(true_utility_vectors))} NaN values in true_utility_vectors")
#     else:
#         print(f"✓ No NaN in true_utility_vectors")
    
#     # Check utilities
#     print("\nTesting utility calculations...")
#     nan_count = 0
#     for i, candidate in enumerate(candidate_vectors):
#         for j, voter in enumerate(voting_vectors):
#             util = utility_function(candidate, voter)
#             if np.isnan(util):
#                 nan_count += 1
#                 if nan_count <= 3:  # Show first 3
#                     print(f"  NaN utility: candidate {i}, voter {j}")
    
#     if nan_count > 0:
#         issues['nan_in_utilities'] = True
#         print(f"❌ Found {nan_count} NaN utilities")
#     else:
#         print(f"✓ All utilities are valid")
    
#     # Check gradients
#     print("\nTesting gradient calculations...")
#     nan_count = 0
#     for i, candidate in enumerate(candidate_vectors):
#         for j, voter in enumerate(voting_vectors):
#             grad = utility_gradient(candidate, voter)
#             if np.any(np.isnan(grad)):
#                 nan_count += 1
#                 if nan_count <= 3:
#                     print(f"  NaN gradient: candidate {i}, voter {j}")
    
#     if nan_count > 0:
#         issues['nan_in_gradients'] = True
#         print(f"❌ Found {nan_count} NaN gradients")
#     else:
#         print(f"✓ All gradients are valid")
    
#     # Check for extreme values
#     print("\nChecking for extreme values...")
#     if np.any(np.abs(candidate_vectors) > 1000):
#         issues['extreme_values'] = True
#         print(f"❌ Extreme values in candidates: max={np.max(np.abs(candidate_vectors))}")
#     else:
#         print(f"✓ No extreme values")
    
#     print("\n" + "=" * 60)
    
#     return issues


# # %%
# # Example 3: Comprehensive analysis across all k values
# print("\n\n" + "=" * 60)
# print("EXAMPLE 3: Comprehensive Analysis Across All K")
# print("=" * 60)

# n_voters = 40
# n_candidates = 6
# d_dimensions = 8
# budget = 12.0
# sparsity = 0.35

# # Generate clustered voters
# true_utility_vectors, voting_vectors, cluster_labels = generate_clustered_voters(
#     n=n_voters,
#     d=d_dimensions,
#     n_clusters=3,
#     cluster_size_variance=0.2,
#     sparsity=sparsity,
#     seed=555
# )

# print(f"Setup: {n_voters} voters, {n_candidates} candidates, {d_dimensions} dimensions")
# print(f"Clusters: {[np.sum(cluster_labels == i) for i in range(3)]}")
# print()

# # Run optimization for all k values
# results_comprehensive = run_simulation_with_optimization_all_k(
#     voting_vectors=voting_vectors,
#     true_utility_vectors=true_utility_vectors,
#     n_candidates=n_candidates,
#     budget=budget,
#     n_simulations=300,
#     learning_rate=0.05,
#     convergence_threshold=1e-2,
#     max_iterations=250,
#     optimization_seed=777,
#     simulation_seed=888,
#     verbose=False  # Set to False for cleaner output
# )

# # Display comprehensive results table
# print("\n" + "=" * 80)
# print("COMPREHENSIVE RESULTS TABLE")
# print("=" * 80)
# print(f"{'k':<5} {'Expected Utility':<20} {'Std Dev':<15} {'Relative to k=1':<20}")
# print("-" * 80)

# baseline_utility = results_comprehensive[1]['expected_utility']
# optimal_k = max(results_comprehensive.keys(), 
#                 key=lambda k: results_comprehensive[k]['expected_utility'])

# for k in range(1, n_candidates + 1):
#     exp_util = results_comprehensive[k]['expected_utility']
#     std_util = results_comprehensive[k]['std_utility']
#     relative_pct = ((exp_util - baseline_utility) / baseline_utility * 100)
    
#     marker = " *** OPTIMAL ***" if k == optimal_k else ""
    
#     print(f"{k:<5} {exp_util:<20.4f} {std_util:<15.4f} {relative_pct:>6.2f}%{marker}")

# print("-" * 80)

# # Statistical analysis
# print("\n" + "=" * 80)
# print("STATISTICAL ANALYSIS")
# print("=" * 80)

# utilities_list = [results_comprehensive[k]['expected_utility'] for k in range(1, n_candidates + 1)]
# max_util = max(utilities_list)
# min_util = min(utilities_list)
# avg_util = np.mean(utilities_list)

# print(f"Maximum utility: {max_util:.4f} (at k={optimal_k})")
# print(f"Minimum utility: {min_util:.4f} (at k={np.argmin(utilities_list) + 1})")
# print(f"Average utility: {avg_util:.4f}")
# print(f"Range: {max_util - min_util:.4f} ({(max_util - min_util)/min_util * 100:.2f}% variation)")

# # Show how utility changes with k
# print("\n" + "=" * 80)
# print("UTILITY TREND")
# print("=" * 80)

# for k in range(1, n_candidates + 1):
#     util = results_comprehensive[k]['expected_utility']
#     bar_length = int((util - min_util) / (max_util - min_util) * 50) if max_util != min_util else 25
#     bar = "█" * bar_length
#     print(f"k={k}: {bar} {util:.2f}")

# # Examine optimized candidates for different k values
# print("\n" + "=" * 80)
# print("OPTIMIZED CANDIDATE STRATEGIES")
# print("=" * 80)

# for k_sample in [1, n_candidates // 2, n_candidates]:
#     print(f"\nk={k_sample} - Optimized Candidates:")
#     for i, candidate in enumerate(results_comprehensive[k_sample]['optimized_candidates']):
#         print(f"  Candidate {i+1}: {np.round(candidate, 2)}")
#         print(f"    Budget: {np.sum(candidate):.4f}, Non-negative dims: {np.sum(candidate > -0.99)}/{d_dimensions}")

# # Compare variance across k
# print("\n" + "=" * 80)
# print("VARIANCE ANALYSIS")
# print("=" * 80)
# print(f"{'k':<5} {'Coefficient of Variation':<25} {'Interpretation':<30}")
# print("-" * 80)

# for k in range(1, n_candidates + 1):
#     exp_util = results_comprehensive[k]['expected_utility']
#     std_util = results_comprehensive[k]['std_utility']
#     cv = (std_util / exp_util * 100) if exp_util > 0 else 0
    
#     if cv < 5:
#         interpretation = "Very stable"
#     elif cv < 10:
#         interpretation = "Stable"
#     elif cv < 20:
#         interpretation = "Moderate variance"
#     else:
#         interpretation = "High variance"
    
#     print(f"{k:<5} {cv:>6.2f}%{'':<18} {interpretation:<30}")

# print("\n" + "=" * 80)


