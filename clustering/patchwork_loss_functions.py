#Loss Functions for the main Patchwork Class
from abc import ABC, abstractmethod
import os
import sys
from math import log2

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


# Define an interface (abstract class) for entropy computation strategies
class LossFunction(ABC):


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
        transition_entropy_loss = patchwork.calculate_entropy_loss(patch,neighbor)
        size_entropy_loss = self.calculate_size_entropy_loss(patch,neighbor)
        
        return transition_entropy_loss + size_entropy_loss
    
    def calculate_size_entropy_loss(self, patchwork, patch, neighbor):
        """
        Calculate the size entropy loss based on the sizes of the patches.
        :param patch: The patch to be merged.
        :param neighbor: The neighboring patch to merge with.
        :return: The size entropy loss.
        """
        rel1 = patchwork.patch_relevances[patch]
        rel2 = patchwork.patch_relevances[neighbor]
        size_entropy_loss = (rel1+rel2)*log2(rel1+rel2) - (rel1*log2(rel1) + rel2*log2(rel2))
        return size_entropy_loss
    
#****************************************************************************************************************