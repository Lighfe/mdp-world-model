import os
import sys
import bisect
from scipy.stats import entropy
import numpy as np
import time
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from datastructures import SortedValueDict
from datasets.data_reconstruction import get_and_reconstruct_data, reconstruct_solver_and_grid
from patchwork_entropy_strategies import *
from patchwork_loss_functions import *
from patchwork_entropy_measures import *

class Patchwork:
    #TODO: Add docstring for the class

    def __init__(self, grid, df, 
                 entropy_strategy: EntropyStrategy = ShannonEntropyOnlyMerged(),
                 loss_function: LossFunction = TransitionEntropyLoss()):
        """
        Initializes the Patchwork object.
        Args:
            grid (Grid): The grid object on which the simulation data is based.
            df (DataFrame): The DataFrame containing the simulation data, already prepared with columns 'x', 'y', 'c'.
            entropy_strategy (EntropyStrategy): Class instance which contains all functions about the transition entropy calculations
            loss_function (LossFunction): Class instanc which contains the calculation of the loss function
        """

        self.entropy_strategy = entropy_strategy
        self.loss_function = loss_function

        #TODO save patches also somehow geometrically?
        self.grid = grid                                        
        self.cell_to_patchindex = {cell: idx for idx, cell in enumerate(self.grid.indices)}     #Dict
        self.patchindex_to_cell = {idx: cell for cell, idx in self.cell_to_patchindex.items()}  #Dict
        self.current_patches = list(range(len(self.grid.indices)))                              #List
        self.patch_neighbors = self._init_patch_nb()                                            #Dict
        #Clustering history                                           
        self.children = dict()                                                                  #Dict
        self.parents = {patch: patch for patch in self.current_patches}                         #Dict    
        self.cell_to_history_of_patches = {cell: [self.cell_to_patchindex[cell]] for cell in self.grid.indices} #Dict                      
        
        self.action_to_control_dict, self.action_df = self._switch_from_control_to_action(df)   #Dict, DataFrame
        self.trans_probs = self._init_tp(self.action_df)                                        #Dict
        self.patch_relevances = self._init_uniform_patch_relevances()                           #Dict

        self.predecessors = self._init_predecessors()                                           #Dict
        self.entropy_dict = self._init_entropy_dict()                                           #Dict
        self.patch_to_history_of_avg_entropy = {patch: [(0, self.entropy_dict[patch]['avg'])] for patch in self.current_patches} #Dict
        self.adjacent_cells_losses = self._init_adjacent_cells_losses()  #SortedValueDict
        
        # Use the patchwork to calculate the initial value(s) of the loss function
        self.loss_function._reset(self)


    def _init_patch_nb(self):
        """
        Initializes a dictionary with the neighbors of each patch.
        """
        cell_nb = self.grid.get_neighbors()
        patch_nb = {self.cell_to_patchindex[cell]: set(self.cell_to_patchindex[n] for n in neighbors) for cell, neighbors in cell_nb.items()}
        return patch_nb      
    
    def _init_tp(self, df):
        """
        Initializes the transition probabilities dictionary by counting the cell transitions in the simulation data.
        
        The dictionary is nested: TP[s][a][s'] = probability of transitioning from s to s' given action a., 
            where s is the current patch, a is the action, and s' is the next patch.

        Args:
            df (DataFrame): The DataFrame containing the simulation data, prepared with columns 'x', 'y', 'a'.
        Returns:
            trans_probs_dict (dict): A nested dictionary TP[s][a][s'] with the transition probabilities.
        """
        #Prepare the data
        df = self._add_cell_and_patch_ids_in_df(df)
        # Count transitions from (x_cell, a) → y_cell
        transition_counts = df.groupby(["x_patch_id", "a"])["y_patch_id"].value_counts().unstack(fill_value=0)
        # Normalize counts to get probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        # Convert to nested dictionary: TP[s][a][s’] = probability 
        trans_probs_dict = {patch: {action: dict() for action in self.action_to_control_dict.keys()} for patch in self.current_patches}
        for (s, a), series in transition_probs.stack().groupby(level=[0, 1]):
            trans_probs_dict[int(s)][a] = {int(keys[2]): prob for keys, prob in series.items() if prob > 0}
       
        return trans_probs_dict 
    
    def _init_uniform_patch_relevances(self):
        """
        Initializes the patch relevances with a uniform distribution over all grid cells.#
        """
        uniform_patch_relevances = {self.cell_to_patchindex[s]: 1/len(self.grid.indices) for s in self.grid.indices}
        return uniform_patch_relevances

    def _init_predecessors(self):
        """
        Initializes a dictionary with the predecessors of each patch based on the transition probabilities.
        Only patches with non-zero probabilities are considered.
        The dictionary is nested: predecessors[s'][a] = [s_1,s_2, ...] with s_i being the patches that reach s' with action a.
        """
        
        predecessors = {patch: {action: set() for action in self.action_to_control_dict.keys()} for patch in self.current_patches}
        for s in self.trans_probs.keys():
            for a in self.trans_probs[s].keys():
                for s_prime in self.trans_probs[s][a].keys():
                    if self.trans_probs[s][a][s_prime] > 0: #should be unnecessary, but for numerical safety (we only want to add predecessors for non-zero probabilities)
                        predecessors[s_prime][a].add(s)
                    else:
                        raise ValueError('There exists a Zero Probability in the Transition Probabilities Dict')
        return predecessors
    
    def _init_entropy_dict(self):
        """
        Initializes a dictionary with the transition entropy of each patch.
        The dictionary is nested: entropy_dict[s][a] = entropy of the probability distribution of action a in patch s.
        Moreover, there is an additional key 'avg' for the average transition entropy of all actions in patch s.
        """
        return self.entropy_strategy._init_entropy_dict(patchwork=self)

    def _init_adjacent_cells_losses(self):
        """
        Initializes a SortedValueDict with the transition entropy loss of merging adjacent patches.
        The dictionary is sorted by the entropy loss and provides efficient functions for:
            - insert(key,value): Inserting a key-value pair.
            - extract_min(): Extracting the key-value pair with the smallest value.
            - remove_by_key(key): Removing the key-value pair with the given key.
        """
        adjacent_cells_losses = SortedValueDict()
        for cell, neighbors in self.patch_neighbors.items():
            for neighbor in neighbors:
                if neighbor > cell: #only calculate once for each pair
                    adjacent_cells_losses.insert((cell, neighbor),self.calculate_loss_of_merging(cell, neighbor))
        return  adjacent_cells_losses 
    
    #****************************************************************************************************************
    #Helper Functions for _init Functions  **************************************************************************
    #****************************************************************************************************************

    def _switch_from_control_to_action(self, df):
        """
        Switches the control column of the df to an action column, where each action is an integer, starting from 0.
        Returns additionally a dictionary mapping actions to controls.

        Args:
            df (DataFrame): The DataFrame containing the simulation data, already prepared with columns 'x', 'y', 'c'.
        Returns:
            action_to_control_dict (dict): A dictionary mapping actions to controls.
            df (DataFrame): The DataFrame with the 'c' column replaced by the 'a' column and the corresponding mapping.
        """
        unique_controls = df['c'].unique()
        action_to_control_dict = {action_idx: control for action_idx, control in enumerate(unique_controls)}
        df['a'] = df['c'].map({control: action_idx for action_idx, control in action_to_control_dict.items()})
        df = df.drop(columns=['c'])
        return action_to_control_dict, df
    
    def _add_cell_and_patch_ids_in_df(self, df):
        """
        Adds columns 'x_cell', 'y_cell', 'x_patch_id', 'y_patch_id' to the DataFrame df.
        'cells' always refer to the grid cells (=their coordinates), 'patch_id' refers to the index of the patch in the list of patches.
        """
        df['x_cell'] = df['x'].apply(self.grid.get_cell_index)
        df['y_cell'] = df['y'].apply(self.grid.get_cell_index)
        df['x_patch_id'] = df['x_cell'].map(self.cell_to_patchindex)
        df['y_patch_id'] = df['y_cell'].map(self.cell_to_patchindex)
        return df
    
    def get_prob_distr_over_all_actions(self, patch):
        ''' Probably won't be used in the final version'''
        combined_probs = dict()
        
        # Sum probabilities from all actions, weighting them equally
        for prob_dist in self.trans_probs[patch].values():
            for y, prob in prob_dist.items():
                combined_probs[y] =combined_probs.get(y, 0) + prob / len(self.trans_probs[patch].keys())  # Weighted sum
        return combined_probs


    #****************************************************************************************************************
    #Functions for the Clustering Algorithm  **************************************************************************
    #****************************************************************************************************************
    
    def compute_merged_entropy(self, patch1,patch2, newpatch = 'new'):
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
        
        return self.entropy_strategy.compute_merged_entropy(self, patch1, patch2, newpatch)
    
    def calculate_loss_of_merging(self, patch1, patch2):
        """
        Calculate the loss incurred by merging two patches.
        This method computes the loss associated with merging two given patches using the specified loss function.
        Args:
            patch1: The first patch to be merged.
            patch2: The second patch to be merged.
        Returns:
            The calculated loss value as a result of merging the two patches.
        """
        
        return self.loss_function.calculate_loss_of_merging(self, patch1, patch2)

    def calculate_entropy_loss(self, patch1, patch2):
        """
        Calculates the relevance-weighted entropy loss of merging patch1 and patch2.
        This is the difference between the entropy of the merged patch and the relevance-weighted sum of the entropies of the two parent patches.
        
        Args: 
                patch1 (int). patch2 (int): The patch indices of the patches to be merged.  
        Returns:
                entropy_loss (float): The relevance-weighted entropy loss of merging patch1 and patch2.
        """

        return self.entropy_strategy.calculate_entropy_loss(self, patch1, patch2)

    def merge_adjacent_patches(self, patch1, patch2, newpatch):
        """
        Merges the two adjacent patches patch1, patch2 into newpatch.
        Update all relevant data structures accordingly.
        """
        
        merged_entropy_dict, merged_probs = self.compute_merged_entropy(patch1, patch2, newpatch)
        self._update_trans_probabilities(patch1, patch2, newpatch, merged_probs)
        self._update_entropy_dict(patch1, patch2, newpatch, merged_entropy_dict)
        patch1_rel, patch2_rel = self._update_patch_relevances(patch1, patch2, newpatch)

        old_predecessors = self._update_predecessors(patch1, patch2, newpatch)
        self._update_patch_neighbors(patch1, patch2, newpatch)
        self._update_adjacent_cells_losses(patch1, patch2, newpatch, old_predecessors)
        self._update_loss_function_value(patch1_rel, patch2_rel)

        self._update_children_and_parents(patch1, patch2, newpatch)
        self._update_cell_to_patch_history(patch1, patch2, newpatch)

        return 
           
    def step(self):
        """
        Performs one step of the clustering algorithm.
        Merges the two adjacent patches with the smallest entropy loss.
        """
        
        patch1, patch2 = self.adjacent_cells_losses.extract_min()[0]
        
        if patch1 is None or patch2 is None:
            print('No more patches to merge')
            return False
        
        newpatch = self.current_patches[-1] + 1
        self.current_patches.append(newpatch)

        self.merge_adjacent_patches(patch1, patch2, newpatch)

        self.current_patches.remove(patch1)
        self.current_patches.remove(patch2)
        
        return True

    def run(self, n_steps=None):
        """
        Runs the clustering algorithm for a given number of steps.
        If n_steps is None, the algorithm runs until no more patches can be merged.
        """
        step = 0
        start_time = time.time()
        while n_steps is None or step < n_steps:
            if step % 100 == 0:
                print(f"Step {step}, number of patches: {len(self.current_patches)}, time: {time.time() - start_time:.2f}")
            if not self.step():
                print(f"Done, after {step} steps")
                break
            step += 1
        
        return

    #****************************************************************************************************************
    #Helper Functions for the Clustering Algorithm  *******************************************************************
    #****************************************************************************************************************

    def _update_trans_probabilities(self, patch1, patch2, newpatch, merged_probs):
        """
        Updates the transition probabilities for merging patch1 and patch2 into newpatch.
        The transition probabilities of patch1 and patch2 are removed from the dictionary.
        
        self.predecessors[s][a] = [s_1, s_2, ...] for each s_i that reaches s with action a.
        """

        # Create newpatch entry
        self.trans_probs[newpatch] = merged_probs

        # Update the transition probabilities of the predecessors of patch1 and patch2, remove old ones
        for patch in [patch1, patch2]:
            for a in self.predecessors[patch]:
                for predecessor in self.predecessors[patch][a]:
                    self.trans_probs[predecessor][a][newpatch] =  self.trans_probs[predecessor][a].get(newpatch,0) + self.trans_probs[predecessor][a][patch]
                    del self.trans_probs[predecessor][a][patch]

        # Remove patch1 and patch2 entries
        del self.trans_probs[patch1]
        del self.trans_probs[patch2]

        return 
    
    def _update_entropy_dict(self, patch1, patch2, newpatch, merged_entropy_dict):
        """
        Updates the entropy dictionary for merging patch1 and patch2 into newpatch.
        The entropies of patch1 and patch2 are removed from the dictionary.
        """

        return self.entropy_strategy._update_entropy_dict(self, patch1, patch2, newpatch, merged_entropy_dict)
    
    def _update_patch_relevances(self, patch1, patch2, newpatch):
        """
        Updates the patch relevances for merging patch1 and patch2 into newpatch.
        The relevance of newpatch is the sum of the relevances of patch1 and patch2.
        The relevances of patch1 and patch2 are removed from the dictionary.
        """
        self.patch_relevances[newpatch] = self.patch_relevances[patch1] + self.patch_relevances[patch2]
        rel1 = self.patch_relevances.pop(patch1)
        rel2 = self.patch_relevances.pop(patch2)
        return rel1, rel2
    
    def _update_predecessors(self, patch1, patch2, newpatch):
        """
        Updates the predecessors dictionary for merging patch1 and patch2 into newpatch.
        The predecessors of newpatch are the union of the predecessors of patch1 and patch2.
        Additionally, the predecessors-set of all patches where patch1 or patch2 are predecessors is updated by replacing patch1 or/and patch2 with newpatch.
        The predecessors of patch1 and patch2 are removed from the dictionary.
        """
        
        old_predecessors = copy.deepcopy(self.predecessors)
        
        # Update all predecessors of newpatch (prior predecessors of patch1 and patch2)
        self.predecessors[newpatch] = dict()
        for patch in [patch1, patch2]:
            for a, patch_predecessors in self.predecessors[patch].items():
                self.predecessors[newpatch].setdefault(a, set()).update(patch_predecessors)

        # Update the predecessors-set of all patches where patch1 or patch2 are predecessors
        for a, action_trans_probs in self.trans_probs[newpatch].items():
            for s_prime in action_trans_probs:
                self.predecessors[s_prime][a].add(newpatch)
                self.predecessors[s_prime][a].discard(patch1) #removes or does nothing
                self.predecessors[s_prime][a].discard(patch2) #removes or does nothing
           
        # Remove patch1 and patch2 entries rom the predecessors dictionary
        self.predecessors.pop(patch1)
        self.predecessors.pop(patch2)
        
        return old_predecessors
    
    def _update_children_and_parents(self, patch1, patch2, newpatch):
        """
        Updates the children and the parents dictionaries for merging patch1 and patch2 into newpatch.
        The children of newpatch are  patch1 and patch2. 
        The parent of patch1 and patch2 is newpatch. Up to now, the parent of newpatch is also newpatch.
        """
        #TODO maybe change to a nested dictionary/list/set of children of children

        self.children[newpatch] = (patch1, patch2)
        self.parents[newpatch] = newpatch
        self.parents[patch1] = newpatch
        self.parents[patch2] = newpatch
        return
    
    def _update_cell_to_patch_history(self, patch1, patch2, newpatch):
        """
        Updates the cell_to_history_of_patches dictionary for merging patch1 and patch2 into newpatch.
        The history of each cell is updated by adding newpatch to every list where patch1 or patch2 are contained.
        """
        #TODO this is rather inefficient, maybe change later or outsorce for visualization
        
        for cell in self.grid.indices:
            if self.cell_to_history_of_patches[cell][-1] in {patch1, patch2}:
                self.cell_to_history_of_patches[cell].append(newpatch)
    
    def _update_patch_neighbors(self, patch1, patch2, newpatch):
        """
        Updates the patch neighbors dictionary for merging patch1 and patch2 into newpatch.
        The neighbors of newpatch are the union of the neighbors of patch1 and patch2 without themselves (patch1 and patch2).
        Additionally, the neighbors-set of all patches where patch1 or patch2 are neighbors is updated by replacing patch1 or/and patch2 with newpatch.
        The entries of patch1 and patch2 are removed from the dictionary.
        """
        
        # Update the neighbors-set of all patches where patch1 or patch2 are neighbors
        for patch in [patch1, patch2]:
            for neighbor in self.patch_neighbors[patch]:
                if neighbor != patch1 and neighbor != patch2:
                    self.patch_neighbors[neighbor].add(newpatch)
                    self.patch_neighbors[neighbor].remove(patch)
            
        # Update all neighbors of newpatch (prior neighbors of patch1 and patch2)
        self.patch_neighbors[newpatch] = self.patch_neighbors[patch1].union(self.patch_neighbors[patch2])
        self.patch_neighbors[newpatch].remove(patch1)
        self.patch_neighbors[newpatch].remove(patch2)

        # Remove patch1 and patch2 entries from the neighbors dictionary
        del self.patch_neighbors[patch1]
        del self.patch_neighbors[patch2]

        return
    
    def _update_adjacent_cells_losses(self, patch1, patch2, newpatch, old_predecessors):

        """
        Updates the adjacent cells losses dictionary for merging patch1 and patch2 into newpatch.
        That is, the entries for the adjacent patches of newpatch are added to the dictionary.
        And the entries which contain patch1 and patch2 are removed from the dictionary.
            (Here the entry for (patch1,patch2) is already removed in the merge_adjacent_patches function.)
        """
        # Find all additional pairs of patches for which the transition entropy loss changes 
        pairs_to_update = self.entropy_strategy._find_pairs_to_update_adjacent_cells_losses(self, patch1, patch2, newpatch, old_predecessors)
        for couple in pairs_to_update:
            self.adjacent_cells_losses.remove_by_key(couple)
            self.adjacent_cells_losses.insert(couple, self.calculate_loss_of_merging(couple[0], couple[1])) #the couple is assumed to be sorted

        # Update separately all adjacent cell losses for direct neighbors of newpatch
        # Here we also need to remove the entries for merging patch1 or patch2 with the neighbor
        for neighbor in self.patch_neighbors[newpatch]:
            # Create newpatch entries
            self.adjacent_cells_losses.insert((neighbor, newpatch) , self.calculate_loss_of_merging(neighbor, newpatch)) #always neighbor < newpatch 
            # Remove all patch1 and patch2 entries
            for patch in [patch1, patch2]:
                key = tuple(sorted((patch, neighbor)))
                self.adjacent_cells_losses.remove_by_key(key)
        return
    
    def _update_loss_function_value(self,  patch1_rel, patch2_rel):
        """
        Updates the loss function value for the current patchwork.
        The loss function value is the sum of the entropy losses.
        """
        self.loss_function.update_values(self, self.entropy_strategy.overall_entropy, patch1_rel, patch2_rel)
        return

    #****************************************************************************************************************
    # Test Functions  ****************************************************************************************
    #****************************************************************************************************************

    def test_adjacent_cells_losses(self, tol = 1e-10):
        """ 
        Test the calculation of the adjacent cell losses.
        """
        #TODO maybe this won't work now as we've substituted the loss value by a tuple
        test_adjacent_cells_losses = SortedValueDict()
        for cell, neighbors in self.patch_neighbors.items():
            for neighbor in neighbors:
                if neighbor > cell: #only calculate once for each pair
                    test_adjacent_cells_losses.insert((cell, neighbor),self.calculate_loss_of_merging(cell, neighbor))
        
        # Compare the two dictionaries
        # Check if the two sorted lists have the same entries
        if set(self.adjacent_cells_losses().keys()) != set(test_adjacent_cells_losses().keys()):
            print("Error: The two dictionaries do not have the same keys.")
            diff_keys = set(self.adjacent_cells_losses.sorted_list).symmetric_difference(set(test_adjacent_cells_losses.sorted_list))
            print(f"Difference in keys: {diff_keys}")
            return False
        
        # Check if the two lists are in the same order
        if [entry[0] for entry in self.adjacent_cells_losses.sorted_list] != [entry[0] for entry in test_adjacent_cells_losses.sorted_list]:           
            diff_order = [(self.adjacent_cells_losses.sorted_list[i], test_adjacent_cells_losses.sorted_list[i]) 
                          for i in range(len(self.adjacent_cells_losses.sorted_list)) 
                          if self.adjacent_cells_losses.sorted_list[i][0] != test_adjacent_cells_losses.sorted_list[i][0]]
            # Maybe the order is different, but the values are the same?
            for pair in diff_order:
                if abs(pair[0][1] - pair[1][1]) > tol:
                    print(f"Error: The two sorted lists do not have the same order.")
                    print(f"Difference in order: {pair[0]} vs {pair[1]}")
                    print(f"Difference also in {diff_order}")
                    return False

        # Check if the two lists have the same values    
        for i in range(len(self.adjacent_cells_losses.sorted_list)):
            if abs(self.adjacent_cells_losses.sorted_list[i][1] - test_adjacent_cells_losses.sorted_list[i][1]) > tol:
                print("Error: The second tuple entries differ beyond the allowed error range.")
                print(f"Difference in values: {self.adjacent_cells_losses.sorted_list[i]} vs {test_adjacent_cells_losses.sorted_list[i]}")
                return False
            
        #print("Test passed: The list of the adjacent cell losses is in the correct order.")
        return  True


    #****************************************************************************************************************
    #Output Functions  ****************************************************************************************
    #****************************************************************************************************************

    def get_cells_to_current_patches(self, step):
        """
        Returns a dictionary mapping each cell to the current patch it belongs to.
        """
        num_grid_cells = len(self.grid.indices)
        cells_to_current_patches = dict()

        for cell, patch_history in self.cell_to_history_of_patches.items():
            latest_patch_with_step = num_grid_cells + step -1
            index = bisect.bisect_left(patch_history, latest_patch_with_step) #binary search for list index in sorted list
            if patch_history[index] == latest_patch_with_step:
                cells_to_current_patches[cell] = patch_history[index]
            else: 
                cells_to_current_patches[cell] = patch_history[index -1]


        return cells_to_current_patches
    
    def get_patches_to_current_avg_entropy(self, step):
        """
        Returns a dictionary mapping each patch to the current average entropy.
        """
        patches_to_current_avg_entropy = dict()

        for patch, avg_entropy_history in self.patch_to_history_of_avg_entropy.items():
            index = bisect.bisect_right([change[0] for change in avg_entropy_history], step) - 1
            patches_to_current_avg_entropy[patch] = avg_entropy_history[index][1]
        return patches_to_current_avg_entropy




def create_patchwork(db_name, 
                     table_name, 
                     run_ids, 
                     entropy_strategy_strg= 'ShannonEntropyOnlyMerged',
                     entropy_measure = 'shannon_entropy',
                     loss_function_strg='TransitionEntropyLoss',
                     loss_function_coeff = None):    
    
    entropy_strategy = globals()[entropy_strategy_strg](globals()[entropy_measure]) #create the entropy strategy object
    loss_function = globals()[loss_function_strg]()
    if loss_function_coeff is not None:
        loss_function.coeff = loss_function_coeff #set the coeffecient of the loss function
           
    #Get the data
    df, configs_dict = get_and_reconstruct_data(db_name, table_name, run_ids)

    # Check if all configurations are the same
    all_equal = all(configs_dict[run_id] == configs_dict[run_ids[0]] for run_id in run_ids)
    print("All configurations are the same:", all_equal)

    grid, solver = reconstruct_solver_and_grid(run_ids[0], configs_dict)
    controls = df['c'].unique() #np.array([np.array(control) for control in df['c'].unique()])

    #Handle finite spaces by throwing out every data which lands outside
    # TODO: make this better, until now we assume that all upper borders are the same and that a space is finite if grid.bounds[0][-1] != np.inf

    if grid.bounds[0][-1] != np.inf:
        rows_to_delete_percentage = df[~df['y'].apply(lambda y: max(y) <= grid.bounds[0][-1])].shape[0] / df.shape[0] * 100
        print(f"Percentage of data rows deleted: {rows_to_delete_percentage:.2f}%")
        df = df[df['y'].apply(lambda y: max(y) <= grid.bounds[0][-1])] #as we run out of the defined space
        patchwork = Patchwork(grid,df, entropy_strategy, loss_function)
    else:
        patchwork = Patchwork(grid, df, entropy_strategy, loss_function)

    return patchwork, controls, solver