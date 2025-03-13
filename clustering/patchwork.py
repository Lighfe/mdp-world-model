import os
import sys
from scipy.stats import entropy
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from datastructures import SortedValueDict

class Patchwork:
    #TODO: Add docstring for the class

    def __init__(self, grid, df):
        """
        Initializes the Patchwork object.
        Args:
            grid (Grid): The grid object on which the simulation data is based.
            df (DataFrame): The DataFrame containing the simulation data, already prepared with columns 'x', 'y', 'c'.
        """

        #TODO save patches also somehow geometrically?
        self.grid = grid                                        
        self.cell_to_patchindex = {cell: idx for idx, cell in enumerate(self.grid.indices)}     #Dict
        self.patchindex_to_cell = {idx: cell for cell, idx in self.cell_to_patchindex.items()}  #Dict
        self.current_patches = list(range(len(self.grid.indices)))                              #List
        self.patch_neighbors = self._init_patch_nb()                                            #Dict                                           
        self.parents = dict()                                                                   #Dict                               
        
        self.action_to_control_dict, action_df = self._switch_from_control_to_action(df)        #Dict, DataFrame
        self.trans_probs = self._init_tp(action_df)                                             #Dict
        self.patch_relevances = self._init_uniform_patch_relevances()                           #Dict

        self.predecessors = self._init_predecessors()                                           #Dict
        self.entropy_dict = self._init_entropy_dict()                                           #Dict
        self.adjacent_cells_losses = self._init_adjacent_cells_losses()                         #SortedValueDict
    
    def _init_patch_nb(self):
        """
        Initializes a dictionary with the neighbors of each patch.
        """
        cell_nb = self.grid.get_neighbors()
        patch_nb = {self.cell_to_patchindex[cell]: [self.cell_to_patchindex[n] for n in neighbors] for cell, neighbors in cell_nb.items()}
        return patch_nb      

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
        trans_probs_dict = {}
        for (s, a), series in transition_probs.stack().groupby(level=[0, 1]):
            trans_probs_dict.setdefault(int(s), {})[a] = {int(keys[2]): prob for keys, prob in series.items() if prob > 0}
       
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
        The dictionary is nested: predecessors[s'] = [(s, a), ...] for each s' that can be reached from s with action a.
        """
        #TODO: Maybe change this to a more nested structure, where the actions are also stored in a middle layer P[s'][a] = [s1,s2, ...]

        predecessors = dict()
        for s in self.trans_probs:
            for a in self.trans_probs[s]:
                for s_prime in self.trans_probs[s][a]:
                    if self.trans_probs[s][a][s_prime] > 0: #should be unnecessary, but for numerical safety (we only want to add predecessors for non-zero probabilities)
                        predecessors.setdefault(s_prime, []).append((s, a))
                    else:
                        raise ValueError('There exists a Zero Probability in the Transition Probabilities Dict')
        return predecessors
    
    def _init_entropy_dict(self):
        """
        Initializes a dictionary with the entropy of each patch.
        The dictionary is nested: entropy_dict[s][a] = entropy of the probability distribution of action a in patch s.
        Moreover, there is an additional key 'avg' for the average entropy of all actions in patch s.
        """
        entropy_dict = dict()
     
        for s in self.current_patches:
            for a in self.trans_probs[s]:
                # Calculate entropy for each action and save it in entropy_dict[s][a]
                entropy_dict.setdefault(s, {})[a] = entropy(list(self.trans_probs[s][a].values()), base=2)
            entropy_dict[s]['avg'] = sum(entropy_dict[s].values()) / len(entropy_dict[s].values())
        
        return entropy_dict
    
    def get_prob_distr_over_all_actions(self, patch):
        ''' Probably won't be used in the final version'''
        combined_probs = dict()
        
        # Sum probabilities from all actions, weighting them equally
        for prob_dist in self.trans_probs[patch].values():
            for y, prob in prob_dist.items():
                combined_probs[y] =combined_probs.get(y, 0) + prob / len(self.trans_probs[patch].keys())  # Weighted sum
        return combined_probs

    
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
        #TODO test if correct!!!! maybe make more efficient

        merged_probs = dict((action, dict()) for action in self.action_to_control_dict.keys())
        merged_entropy_dict = dict()

        for a in self.action_to_control_dict.keys():
            # For each action, sum probabilities from both patches, weighting them bei patch_relevances
            for s in [patch1,patch2]:
                for s_prime, prob in self.trans_probs[s][a].items():
                    if s_prime != patch1 and s_prime != patch2:
                        merged_probs[a][s_prime] = merged_probs[a].get(s_prime,0) + self.patch_relevances[s]*prob
                    else:
                        merged_probs[a][newpatch] = merged_probs[a].get(newpatch,0) + self.patch_relevances[s]*prob
            # For each action, calculate the entropy 
            merged_entropy_dict[a] = entropy(list(merged_probs[a].values()), base = 2)
        
        # Calculate the average of the action entropies
        merged_entropy_dict['avg'] = sum(merged_entropy_dict.values()) / len(merged_entropy_dict)

        return merged_entropy_dict, merged_probs

    

    def calculate_entropy_loss(self, patch1, patch2):
        """
        Calculates the relevance-weighted entropy loss of merging patch1 and patch2.
        This is the difference between the entropy of the merged patch and the relevance-weighted sum of the entropies of the two parent patches.
        
        Args: 
                patch1 (int). patch2 (int): The patch indices of the patches to be merged.  
        Returns:
                entropy_loss (float): The relevance-weighted entropy loss of merging patch1 and patch2.
        """
        relevance1, avg_entropy1 = self.patch_relevances[patch1], self.entropy_dict[patch1]['avg']
        relevance2, avg_entropy2 = self.patch_relevances[patch2], self.entropy_dict[patch2]['avg']

        merged_entropy, merged_probs = self.compute_merged_entropy(patch1,patch2)

        entropy_loss = (relevance1 + relevance2) * merged_entropy['avg'] - (relevance1 * avg_entropy1 + relevance2 * avg_entropy2)

        return entropy_loss


    def _init_adjacent_cells_losses(self):
        """
        Initializes a SortedValueDict with the entropy loss of merging adjacent patches.
        The dictionary is sorted by the entropy loss and provides efficient functions for:
            - insert(key,value): Inserting a key-value pair.
            - extract_min(): Extracting the key-value pair with the smallest value.
            - remove_by_key(key): Removing the key-value pair with the given key.
        """
        adjacent_cells_losses = SortedValueDict()
        for cell, neighbors in self.patch_neighbors.items():
            for neighbor in neighbors:
                if neighbor > cell: #only calculate once for each pair
                    adjacent_cells_losses.insert((cell, neighbor),self.calculate_entropy_loss(cell, neighbor))
        return  adjacent_cells_losses 
        
