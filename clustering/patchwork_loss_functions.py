#Loss Functions for the main Patchwork Class
from abc import ABC, abstractmethod
import os
import sys
from math import log2

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


# Define an interface (abstract class) for entropy computation strategies
class LossFunction(ABC):

    def _reset(self, patchwork):
        """
        Reset the loss function attributes to the actual current values.
        Needed for some loss functions that depend on the complete current state of the patchwork.
        :param patchwork: The patchwork object containing the patches.
        """
        pass
    
    def _update(self, patchwork, patch, neighbor):
        """
        Update the loss function attributes based on the current state of the patchwork and the patches which are merged.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
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

    def calculate_loss_of_merging(self, patchwork, patch, neighbor):
        """
        Calculate the loss of merging two patches based on transition entropy.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The loss of merging the two patches.
        """
        
        return patchwork.calculate_entropy_loss(patch,neighbor)
    



#****************************************************************************************************************

class TransitionAndSizeEntropyLoss(LossFunction):
    """
    Loss function based on transition and size entropy.
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

    def _reset(self, patchwork):
        """
        Reset the current size entropy to the actual size entropy.
        :param patchwork: The patchwork object containing the patches.
        """
        self.current_size_entropy = self.calculate_size_entropy(patchwork)
        print(f"Reset current size entropy to {self.current_size_entropy}")

    def _update_adjacent_cells_losses(self, patchwork, patch, neighbor, newpatch, rel1, rel2):
        """
        Update the current size entropy based on the current state of the patchwork and the patches which are merged.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        """
        for couple, loss  in patchwork.adjacent_cells_losses().items():
            pass
        
        self.current_size_entropy = self.calculate_next_size_entropy(patchwork, patch, neighbor, rel1, rel2)
        

    def calculate_size_entropy(self, patchwork):
        """
        Calculate the size entropy based on the sizes of the patches.
        :param patchwork: The patchwork object containing the patches.
                            Assuming the patch_relevances are normalized to (patch_size / total_size)!
        :return: The size entropy.
        """
        size_entropy = 1 - (sum(patchwork.patch_relevances[patch] * log2(patchwork.patch_relevances[patch]) for patch in patchwork.patches) / log2(len(patchwork.current_patches)))
        return size_entropy

    def calculate_loss_of_merging(self, patchwork, patch, neighbor):
        """
        Calculate the loss of merging two patches based on transition entropy.
        :param patchwork: The patchwork object containing the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The loss of merging the two patches.
        """
        transition_entropy_loss = patchwork.calculate_entropy_loss(patch,neighbor)
        size_entropy_loss = self.calculate_size_entropy_loss(patch,neighbor)

        return transition_entropy_loss + size_entropy_loss
    
    def calculate_next_size_entropy(self, patch, neighbor, patchwork, rel1 = None, rel2 = None):
        """
        Calculate the next size entropy based on the sizes of the patches.
        
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :param patchwork: The patchwork object containing the patches.
        :return: The next size entropy.        
        """
        if rel1 is None or rel2 is None:
            # If the relevances are not provided, get them from the patchwork
            rel1 = patchwork.patch_relevances[patch]
            rel2 = patchwork.patch_relevances[neighbor]
        n_patches = len(patchwork.current_patches)
        old_kernel = (self.current_size_entropy - 1)*log2(n_patches)
        new_kernel = old_kernel + rel1*log2(rel1) + rel2*log2(rel2) - (rel1+rel2)*log2(rel1+rel2)
        new_size_entropy = (new_kernel / log2(n_patches - 1)) + 1
        return new_size_entropy
    
    def calculate_size_entropy_loss(self, patchwork, patch, neighbor):
        """
        Calculate the size entropy loss based on the sizes of the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The size entropy loss, the smaller the better.
        """
        size_entropy_loss = self.calculate_next_size_entropy(patch, neighbor, patchwork) - self.current_size_entropy 
        return size_entropy_loss
    
#****************************************************************************************************************