from scipy.stats import entropy


def shannon_entropy(probs_dict, source_patch):
    """
    Calculate the Shannon entropy of a source patch based on the probabilities of transitions.

    Args:
        probs_dict (dict): Dictionary containing probabilities for each patch which is reached.
                            keys: target patches, values: probability of transition from source patch to that target patch.
        source_patch (str): The source patch for which to calculate the entropy. May be a key in probs_dict.

    Returns:
        float: The Shannon entropy of the source patch.
    """
    
    return entropy(list(probs_dict.values()), base=2)




def conditional_shannon_entropy(probs_dict, source_patch):
    """
    Calculate the conditional Shannon entropy of a source patch based on the probabilities of transitions.

    Args:
        probs_dict (dict): Dictionary containing probabilities for each patch which is reached.
                            keys: target patches, values: probability of transition from source patch to that target patch.
        source_patch (str): The source patch for which to calculate the conditional entropy. May be a key in probs_dict.

    Returns:
        float: The conditional Shannon entropy of the source patch.
    """
    
    probs = [v for k, v in probs_dict.items() if k != source_patch]
    sum_probs = sum(probs)
    if sum_probs > 0:
        probs = [p / sum_probs for p in probs]
    elif sum_probs == 0:
        return 0.0
    else:
        raise ValueError(f"Sum of transition probabilities cannot be negative (here for patch {source_patch}).")
    return entropy(probs, base=2)