import numpy as np
from scipy.stats import entropy

def calculate_entropy(data, base=2):
    """
    Calculate entropy of a dataset
    Params: data: np.array(n_samples, n_dim)
            base: int, base of the logarithm of the entropy
    Returns: entropy_value : float, entropy of the dataset
             unique: np.array(n_unique_states, n_dim), unique states in the dataset, e.g. all grid cells contained
             probabilities: np.array(n_unique_states), probabilities of each unique state  
    """
    
    flattened = [tuple(row) for row in data]
    # Count occurrences of each unique state and calculate probabilities
    unique, counts = np.unique(flattened, axis=0, return_counts=True)
    probabilities = counts / len(flattened)
    # Calculate entropy using scipy
    entropy_value = entropy(probabilities, base=base)
    
    return entropy_value, unique, probabilities


def create_stopping_by_entropy_threshold_criteria(threshold, base=2):
    def stopping_by_entropy_threshold(data):
        """
        Stopping criteria based on the overall entropy of the dataset
        Params: data: np.array(num_steps,n_samples, n_dim)
        Returns: bool, True if the entropy of the dataset is below or equal the threshold
        """
        data = data.reshape(-1, data.shape[-1])
        entropy_value, unique, probabilities = calculate_entropy(data, base)
        if entropy_value <= threshold:
            print(f"Stopped at entropy: {entropy_value} <= {threshold} (threshold)")
        return entropy_value <= threshold
    return stopping_by_entropy_threshold


def entropy_of_uniform_distribution_over_percentage(n, percentage, base=2):
    """
    Calculate the entropy of a uniform distribution over a percentage of the states
    Params: n: int, total number of states
            percentage: float, percentage of states to consider
            base: int, base of the logarithm of the entropy
    Returns: entropy_value : float, entropy of the uniform distribution
    """
    probabilities = np.zeros(n)
    uniover = int(np.ceil(percentage * n))
    probabilities[:uniover] = 1 / uniover
    # Normalize to make sure the total probability is 1
    probabilities /= probabilities.sum()
    
    return entropy(probabilities, base=base)