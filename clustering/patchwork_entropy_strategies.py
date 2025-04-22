#Strategy Patterns for the Entropy Calculation in the main Patchwork Class
from abc import ABC, abstractmethod
import os
import sys
import bisect
import copy
from scipy.stats import entropy
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)


# Define an interface (abstract class) for entropy computation strategies
class EntropyStrategy(ABC):

    def __init__(self):
        self.overall_entropy = 0
    
    
    def _init_entropy_dict(self, patchwork):
        """
        Initializes a dictionary with the entropy of each patch.
        The dictionary is nested: entropy_dict[s][a] = entropy of the probability distribution of action a in patch s.
        Moreover, there is an additional key 'avg' for the average entropy of all actions in patch s.
        """
        # Initialize the entropy dictionary with the entropy of each patch
        entropy_dict = dict()
     
        for s in patchwork.current_patches:
            for a in patchwork.trans_probs[s]:
                # Calculate entropy for each action and save it in entropy_dict[s][a]
                entropy_dict.setdefault(s, {})[a] = entropy(list(patchwork.trans_probs[s][a].values()), base=2)
            entropy_dict[s]['avg'] = sum(entropy_dict[s].values()) / len(entropy_dict[s].values())
            self.overall_entropy += patchwork.patch_relevances[s] * entropy_dict[s]['avg']
        
        return entropy_dict
    

    def compute_merged_entropy(self, patchwork, patch1, patch2, newpatch='new'):
        """
        Computes the entropy of the merged patch when merging patch1 and patch2.
        This is done by:
            1. the transition probabilities of the merged patch are calculated from the transition probabilities of patch1 and patch2, weighted by the patch relevances.
            2. the entropy of the merged patch is calculated from the transition probabilities of the merged patch.
        
        Args: 
            patch1 (int). patch2 (int): The patch indices of the patches to be merged.
            newpatch (hashable type): The index of the new patch that is created by merging patch1 and patch2.
        Returns:
            merged_entropy_dict (dict): A dictionary with the entropy of the merged patch for each action, as well as for ['avg'] the average entropy over all actions.
            merged_probs (dict): A dictionary with the transition probabilities of the merged patch for each action.
        """
        #TODO test if correct!!!! maybe make more efficient

        merged_probs = dict((action, dict()) for action in patchwork.action_to_control_dict.keys())
        merged_entropy_dict = dict()
        total_patch_relevance = patchwork.patch_relevances[patch1] + patchwork.patch_relevances[patch2]

        for a in patchwork.action_to_control_dict.keys():
            # For each action, sum probabilities from both patches, weighting them bei patch_relevances
            for s in [patch1,patch2]:
                for s_prime, prob in patchwork.trans_probs[s][a].items():
                    if s_prime != patch1 and s_prime != patch2:
                        merged_probs[a][s_prime] = merged_probs[a].get(s_prime,0) + (patchwork.patch_relevances[s] / total_patch_relevance) *prob
                    else:
                        merged_probs[a][newpatch] = merged_probs[a].get(newpatch,0) + (patchwork.patch_relevances[s] / total_patch_relevance) *prob
            # For each action, calculate the entropy 
            merged_entropy_dict[a] = entropy(list(merged_probs[a].values()), base = 2)
        
        # Calculate the average of the action entropies
        merged_entropy_dict['avg'] = sum(merged_entropy_dict.values()) / len(merged_entropy_dict)

        return merged_entropy_dict, merged_probs



    @abstractmethod
    def calculate_entropy_loss(self, patchwork, patch1, patch2):
        """
        Calculates the relevance-weighted entropy loss of merging patch1 and patch2.
        This is the difference between the entropy of the merged patch and the relevance-weighted sum of the entropies of the two parent patches.
        
        Args: 
                patch1 (int). patch2 (int): The patch indices of the patches to be merged.  
        Returns:
                entropy_loss (float): The relevance-weighted entropy loss of merging patch1 and patch2.
        """
        pass

    @abstractmethod
    def _update_entropy_dict(self, patchwork, patch1, patch2, newpatch, merged_entropy_dict):
        """
        Updates the entropy dictionary for merging patch1 and patch2 into newpatch.
        """
        pass

    @abstractmethod
    def _update_adjacent_cells_losses(self, patchwork, patch1, patch2, newpatch, old_predecessors):

        """
        Updates the adjacent cells losses dictionary for merging patch1 and patch2 into newpatch.
        That is, the entries for the adjacent patches of newpatch are added to the dictionary.
        And the entries which contain patch1 and patch2 are removed from the dictionary.
            (Here the entry for (patch1,patch2) is already removed in the merge_adjacent_patches function.)
        """
        pass
    

#********************************************************************************************************************
#******************* Concrete strategies for the entropy calculation *******************
#********************************************************************************************************************

class ShannonEntropyOnlyMerged(EntropyStrategy):
    """
    This strategy computes the Shannon entropy of the merged patch.
    It does not take into account the entropies of the parent patches in the entropy loss calculation.
    """
    
            
    def calculate_entropy_loss(self, patchwork, patch1, patch2):
        """
        Calculates the relevance-weighted entropy loss of merging patch1 and patch2.
        This is the difference between the entropy of the merged patch and the relevance-weighted sum of the entropies of the two parent patches.
        
        Args: 
                patch1 (int). patch2 (int): The patch indices of the patches to be merged.  
        Returns:
                entropy_loss (float): The relevance-weighted entropy loss of merging patch1 and patch2.
        """
        relevance1, avg_entropy1 = patchwork.patch_relevances[patch1], patchwork.entropy_dict[patch1]['avg']
        relevance2, avg_entropy2 = patchwork.patch_relevances[patch2], patchwork.entropy_dict[patch2]['avg']

        merged_entropy, merged_probs = self.compute_merged_entropy(patchwork, patch1,patch2)
        

        entropy_loss = (relevance1 + relevance2) * merged_entropy['avg'] - (relevance1 * avg_entropy1 + relevance2 * avg_entropy2)

        return entropy_loss
    
    def _update_entropy_dict(self, patchwork, patch1, patch2, newpatch, merged_entropy_dict):
        """
        Updates the entropy dictionary for merging patch1 and patch2 into newpatch.
        The entropies of patch1 and patch2 are removed from the dictionary.
        """
        patchwork.entropy_dict[newpatch] = merged_entropy_dict
        step = newpatch + 1 - len(patchwork.grid.indices)
        patchwork.patch_to_history_of_avg_entropy[newpatch] = [(step, merged_entropy_dict['avg'])]

        # Update the overall entropy
        self.overall_entropy += (patchwork.patch_relevances[patch1] + patchwork.patch_relevances[patch2]) * merged_entropy_dict['avg']
        self.overall_entropy -= patchwork.patch_relevances[patch1] * patchwork.entropy_dict[patch1]['avg']
        self.overall_entropy -= patchwork.patch_relevances[patch2] * patchwork.entropy_dict[patch2]['avg']

        # Remove the entropies of patch1 and patch2 from the dictionary
        del patchwork.entropy_dict[patch1]
        del patchwork.entropy_dict[patch2]

        return
    
    def _update_adjacent_cells_losses(self, patchwork, patch1, patch2, newpatch, old_predecessors):
        """
        Updates the adjacent cells losses dictionary for merging patch1 and patch2 into newpatch.
        That is, the entries for the adjacent patches of newpatch are added to the dictionary.
        And the entries which contain patch1 and patch2 are removed from the dictionary.
            (Here the entry for (patch1,patch2) is already removed in the merge_adjacent_patches function.)
        """
        
        for neighbor in patchwork.patch_neighbors[newpatch]:
            # Create newpatch entries
            patchwork.adjacent_cells_losses.insert((neighbor, newpatch) , patchwork.calculate_loss_of_merging(neighbor, newpatch)) #always neighbor < newpatch 
            # Remove all patch1 and patch2 entries
            for patch in [patch1, patch2]:
                key = tuple(sorted((patch, neighbor)))
                patchwork.adjacent_cells_losses.remove_by_key(key)
        return



class ShannonEntropyAll(EntropyStrategy):
    """
    This strategy computes the Shannon entropy of the merged patch.
    It takes into account the changing entropies of the predecessor patches in the entropy loss calculation.
    """

    def _get_common_predecessors_of_2_patches(self, patchwork, patch1, patch2):
        """
        Finds the predecessors of patch1 and patch2.
        """
        predecessors1 = set().union(*[preds for preds in patchwork.predecessors[patch1].values()])
        predecessors2 = set().union(*[preds for preds in patchwork.predecessors[patch2].values()])
        common_predecessors = predecessors1.intersection(predecessors2)
        common_predecessors.discard(patch1)
        common_predecessors.discard(patch2)
        
        return common_predecessors
    
    def _compute_trans_probs_of_newpatch_predecessors(self, patchwork, predecessor, patch1, patch2, newpatch):

        new_trans_probs = copy.deepcopy(patchwork.trans_probs[predecessor])
        for a in new_trans_probs:
            new_trans_probs[a][newpatch] = new_trans_probs[a].get(patch1, 0) + new_trans_probs[a].get(patch2, 0)
            new_trans_probs[a].pop(patch1, None)
            new_trans_probs[a].pop(patch2, None)

        return new_trans_probs


    def _compute_entropy_dict_from_trans_probs(self, trans_probs):
        entropy_dict = dict()
        for a in trans_probs:
            entropy_dict[a] = entropy(list(trans_probs[a].values()), base=2)
        entropy_dict['avg'] = sum(entropy_dict.values()) / len(entropy_dict.values())
        return entropy_dict

    
            
    def calculate_entropy_loss(self, patchwork, patch1, patch2):
        """
        Calculates the relevance-weighted entropy loss of merging patch1 and patch2.
        This is the difference between the entropy of the merged patch and the relevance-weighted sum of the entropies of the two parent patches.
        
        Args: 
                patch1 (int). patch2 (int): The patch indices of the patches to be merged.  
        Returns:
                entropy_loss (float): The relevance-weighted entropy loss of merging patch1 and patch2.
        """
        relevance1, avg_entropy1 = patchwork.patch_relevances[patch1], patchwork.entropy_dict[patch1]['avg']
        relevance2, avg_entropy2 = patchwork.patch_relevances[patch2], patchwork.entropy_dict[patch2]['avg']

        merged_entropy, merged_probs = self.compute_merged_entropy(patchwork, patch1,patch2)
        
        entropy_loss_direct = (relevance1 + relevance2) * merged_entropy['avg'] - (relevance1 * avg_entropy1 + relevance2 * avg_entropy2)

        # Calculate the entropy loss for the common predecessor patches
        predecessors = self._get_common_predecessors_of_2_patches(patchwork, patch1, patch2)
        predecessor_entropy_losses = [] 
        
        for pred in predecessors:
            current_entropy = patchwork.entropy_dict[pred]['avg']
            new_trans_probs = self._compute_trans_probs_of_newpatch_predecessors(patchwork, pred, patch1, patch2, 'new')
            new_entropy_dict = self._compute_entropy_dict_from_trans_probs(new_trans_probs)
            predecessor_entropy_losses.append((patchwork.patch_relevances[pred] * (new_entropy_dict['avg'] - current_entropy)))

        entropy_loss_total = entropy_loss_direct + sum(predecessor_entropy_losses)    
        
        return entropy_loss_total
    

    def _update_entropy_dict(self, patchwork, patch1, patch2, newpatch, merged_entropy_dict):
        """
        Updates the entropy dictionary for merging patch1 and patch2 into newpatch.
        """
        patchwork.entropy_dict[newpatch] = merged_entropy_dict
        step = newpatch + 1 - len(patchwork.grid.indices)
        patchwork.patch_to_history_of_avg_entropy[newpatch] = [(step, merged_entropy_dict['avg'])]

        # Update the overall entropy
        self.overall_entropy += (patchwork.patch_relevances[patch1] + patchwork.patch_relevances[patch2]) * merged_entropy_dict['avg']
        self.overall_entropy -= patchwork.patch_relevances[patch1] * patchwork.entropy_dict[patch1]['avg']
        self.overall_entropy -= patchwork.patch_relevances[patch2] * patchwork.entropy_dict[patch2]['avg']
        
        # Update the entropies of the common predecessors of patch1 and patch2
        predecessors = self._get_common_predecessors_of_2_patches(patchwork, patch1, patch2)
        for pred in predecessors:
            self.overall_entropy -= patchwork.patch_relevances[pred] * patchwork.entropy_dict[pred]['avg']
            patchwork.entropy_dict[pred] = self._compute_entropy_dict_from_trans_probs(patchwork.trans_probs[pred])
            self.overall_entropy += patchwork.patch_relevances[pred] * patchwork.entropy_dict[pred]['avg']
            patchwork.patch_to_history_of_avg_entropy[pred].append((step, merged_entropy_dict['avg']))

        return
    
    def _update_adjacent_cells_losses(self, patchwork, patch1, patch2, newpatch, old_predecessors):
        """
        Updates the adjacent cells losses dictionary for merging patch1 and patch2 into newpatch.
        That is, the entries for the adjacent patches of newpatch are added to the dictionary.
        And the entries which contain patch1 and patch2 are removed from the dictionary.
            (Here the entry for (patch1,patch2) is already removed in the merge_adjacent_patches function.)
        """
        # Update all adjacent cell losses for predecessors and their neighbors
        # The entropy only changes if both former patches are reached by the predecessor and its neighbor 

        # Get the predecessors of patch1 and patch2
        predecessors1 = set().union(*[preds for preds in old_predecessors[patch1].values()])
        predecessors2 = set().union(*[preds for preds in old_predecessors[patch2].values()])
        common_predecessors = predecessors1.intersection(predecessors2)
        predecessors_only1 = predecessors1.difference(predecessors2)
        predecessors_only2 = predecessors2.difference(predecessors1)
        # Remove patch1 and patch2 from all three sets
        for patch in [patch1, patch2]:
            common_predecessors.discard(patch)
            predecessors_only1.discard(patch)
            predecessors_only2.discard(patch)

        pairs_to_update = set()
        for pred in common_predecessors:
            for neighbor in patchwork.patch_neighbors[pred]:
                if neighbor != newpatch:
                    pairs_to_update.add(tuple(sorted((pred, neighbor))))
        for pred in predecessors_only1:
            for neighbor in patchwork.patch_neighbors[pred]:
                if neighbor != newpatch and neighbor in predecessors_only2:
                    pairs_to_update.add(tuple(sorted((pred, neighbor))))


        #Consider couples of cells of which newpatch is a common predecessor
        newpatch_targets = set().union([targetpatch 
                                         for actiondict in patchwork.trans_probs[newpatch].values() 
                                         for targetpatch in actiondict.keys()
                                         if targetpatch != newpatch])
        #find all neighbored pairs in newpatch_targets
        newpatch_targets = sorted(list(newpatch_targets))  # Convert to list for indexing
        for i, target1 in enumerate(newpatch_targets):
            for target2 in newpatch_targets[i + 1:]:
                if target2 in patchwork.patch_neighbors[target1]:
                    pairs_to_update.add((target1, target2)) #is sorted because of sorting before

        #print("pairs_to_update", len(pairs_to_update))
        for couple in pairs_to_update:
            # TODO check how many are actually changing something
            old_loss = patchwork.adjacent_cells_losses()[couple]
            patchwork.adjacent_cells_losses.remove_by_key(couple)
            new_loss = patchwork.calculate_loss_of_merging(couple[0], couple[1])
            patchwork.adjacent_cells_losses.insert(couple, new_loss)
            if new_loss != old_loss:
                #print(f"changed loss {new_loss-old_loss} for {couple}")
                pass

        # Update separately all adjacent cell losses for direct neighbors of newpatch
        # Here we also need to remove the entries for merging patch1 or patch2 with the neighbor
        for neighbor in patchwork.patch_neighbors[newpatch]:
            # Create newpatch entries
            patchwork.adjacent_cells_losses.insert((neighbor, newpatch) , patchwork.calculate_loss_of_merging(neighbor, newpatch)) #always neighbor < newpatch 
            # Remove all patch1 and patch2 entries
            for patch in [patch1, patch2]:
                key = tuple(sorted((patch, neighbor)))
                patchwork.adjacent_cells_losses.remove_by_key(key)


        return
