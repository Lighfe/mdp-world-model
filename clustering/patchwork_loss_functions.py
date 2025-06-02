#Loss Functions for the main Patchwork Class
from abc import ABC, abstractmethod
import os
import sys
from math import log2

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


# Define an interface (abstract class) for entropy computation strategies
class LossFunction(ABC):

    def __init__(self):
        self.current_total_loss_function_value = 0 
        self.history_of_loss_function_values = {"Loss Function Value": []}
    
    @abstractmethod
    def _reset(self, patchwork):
        """
        Reset the loss function attributes to the actual current values.
        Needed for some loss functions that depend on the complete current state of the patchwork.
        :param patchwork: The patchwork object containing the patches.
        """
        pass
    
    @abstractmethod
    def update_values(self, patchwork, total_transition_entropy, patch1_rel, patch2_rel):
        """
        Update the entropy values after merging patch1 and patch2.
        Also, update the history of the loss function values.
        :param patchwork: The patchwork object containing the patches.
        :param total_transition_entropy: The total transition entropy of the patchwork.
        :param patch1_rel: The relevance of the first patch.
        :param patch2_rel: The relevance of the second patch.
        """
        pass

    @abstractmethod
    def calculate_loss_of_merging(self, patchwork, patch, neighbor):
        """
        Calculate the loss of merging two patches.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The loss of merging the two patches.
        """
        pass


 #****************************************************************************************************************
#*****************************************************************************************************************

class TransitionEntropyLoss(LossFunction):
    """
    Loss function based on transition entropy.
    """

    def __init__(self):
        self.current_transition_entropy = 0
        self.history_of_loss_function_values = {"Total Transition Entropy": [], "Loss Function Value": []}
        self.loss_function_strg = f"Loss_fct = Total Transition Entropy"

    @property
    def current_total_loss_function_value(self):
        return self.current_transition_entropy

    def _reset(self, patchwork):
        """
        Reset the current_transition_entropy and thus also the total loss_function_value
        :param patchwork: The patchwork object containing the patches.
        """
        self.current_transition_entropy = patchwork.entropy_strategy.overall_entropy
        self.history_of_loss_function_values["Total Transition Entropy"].append(self.current_transition_entropy)
        self.history_of_loss_function_values["Loss Function Value"].append(self.current_total_loss_function_value)


    def update_values(self, patchwork, total_transition_entropy, patch1_rel, patch2_rel):
        """
        Update the current size entropy and transition entropy after merging patch1 and patch2.
        :param patchwork: The patchwork object containing the patches.
        :param total_transition_entropy: The total transition entropy of the patchwork.
        :param patch1_rel: The relevance of the first patch.
        :param patch2_rel: The relevance of the second patch.
        """
        self.current_transition_entropy = total_transition_entropy
        self.history_of_loss_function_values["Total Transition Entropy"].append(self.current_transition_entropy)
        self.history_of_loss_function_values["Loss Function Value"].append(self.current_total_loss_function_value)
        return


    def calculate_loss_of_merging(self, patchwork, patch, neighbor):
        """
        Calculate the loss of merging two patches based on transition entropy.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The loss of merging the two patches.
        """
        entropy_loss = patchwork.calculate_entropy_loss(patch,neighbor)
        total_loss = entropy_loss
        return (total_loss, entropy_loss)
    



#****************************************************************************************************************

class TransitionAndSizeEntropyLoss(LossFunction):
    """
    Loss function based on transition and size entropy.

    Loss function = total average transition entropy - size entropy

    The size entropy is calculated based on the sizes of the patches: 
        size_entropy = H(Größenverteilung) 
        #### not anymore 1 - (H(Größenverteilung) / log2(number of patches)), because of better updating
    where H(Größenverteilung) is the entropy of the size distribution of the patches,
        with 1/patch_size being interpreted as the probability of a patch
        with the patch_size being normalized to (patch_size / total_size).
        For the patch_size, we use the patch_relevance.

    As the size entropy depends on all patches, we need to store and update it.
    """

    def __init__(self):
        self.current_size_entropy = 0
        self.current_transition_entropy = 0
        self.history_of_loss_function_values = {"Total Size Entropy": [], "Total Transition Entropy": [], "Loss Function Value": []}
        self.coeff = 1

    @property
    def current_total_loss_function_value(self):
        return self.current_transition_entropy - self.current_size_entropy 
    
    @property
    def loss_function_strg(self):
        return f"Loss_fct = Total Trans. H - {self.coeff}* H(size distr.)"

    def _reset(self, patchwork):
        """
        Reset the current size and transition entropies to the actual values.
        Reset the history of size and transition entropies.
        :param patchwork: The patchwork object containing the patches.
        """
        self.current_size_entropy = self.calculate_size_entropy(patchwork)
        self.current_transition_entropy = patchwork.entropy_strategy.overall_entropy
        # reset the history of size and transition entropies
        self.history_of_loss_function_values["Total Size Entropy"] = [self.current_size_entropy]
        self.history_of_loss_function_values["Total Transition Entropy"] = [self.current_transition_entropy]
        self.history_of_loss_function_values["Loss Function Value"] = [self.current_total_loss_function_value]

    def update_values(self, patchwork, total_transition_entropy, patch1_rel, patch2_rel):
        """
        Update the current size entropy and transition entropy after merging patch1 and patch2.
        :param patchwork: The patchwork object containing the patches.
        :param total_transition_entropy: The total transition entropy of the patchwork.
        :param patch1_rel: The relevance of the first patch.
        :param patch2_rel: The relevance of the second patch.
        """
        self.current_size_entropy = self.update_size_entropy(patch1_rel, patch2_rel)
        self.current_transition_entropy = total_transition_entropy
        # update the history of size and transition entropies
        self.history_of_loss_function_values["Total Size Entropy"].append(self.current_size_entropy)
        self.history_of_loss_function_values["Total Transition Entropy"].append(self.current_transition_entropy)
        self.history_of_loss_function_values["Loss Function Value"].append(self.current_total_loss_function_value)
        return

    def calculate_size_entropy(self, patchwork):
        """
        Calculate the size entropy based on the sizes of the patches.
        :param patchwork: The patchwork object containing the patches.
                            Assuming the patch_relevances are normalized to (patch_size / total_size)!
        :return: The size entropy.
        """
        size_entropy = - self.coeff * sum(patchwork.patch_relevances[patch] * log2(patchwork.patch_relevances[patch]) for patch in patchwork.current_patches) 
        return size_entropy
    
    
    def calculate_size_entropy_loss(self, rel1, rel2):
        """
        Calculate the size entropy loss based on the sizes of the patches.
        :param rel1: The relative size of the first patch.
        :param rel2: The relative size of the second patch.
        :return: The size entropy loss, the smaller the better.
        """
        size_entropy_loss = self.coeff*((rel1+rel2)*log2(rel1+rel2) - rel1*log2(rel1) - rel2*log2(rel2))
        return size_entropy_loss
    

    def calculate_loss_of_merging(self, patchwork, patch, neighbor):
        """
        Calculate the loss of merging two patches based on transition entropy.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The loss of merging the two patches.
        """
        transition_entropy_loss = patchwork.calculate_entropy_loss(patch,neighbor) 
        size_entropy_loss = self.calculate_size_entropy_loss(patchwork.patch_relevances[patch], patchwork.patch_relevances[neighbor]) #the bigger the loss, the smaller the entropy
        total_loss = transition_entropy_loss + size_entropy_loss 
        # because we want to minimize both (for different reasons)
        return (total_loss, transition_entropy_loss, size_entropy_loss)
    

    def update_size_entropy(self, rel1, rel2):
        """
        Calculate the next size entropy based on the sizes of the patches.
        :param rel1: The relative size of the first patch.
        :param rel2: The relative size of the second patch.
        :return: The next size entropy.        
        """
        return self.current_size_entropy - self.calculate_size_entropy_loss(rel1,rel2)
    
#****************************************************************************************************************