import os
import sys
from scipy.stats import entropy
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)



class Patchwork:
    

    def __init__(self, grid, df):
        #TODO save patches also somehow geometrically?
        self.grid = grid
        self.df, self.cell_to_patchindex = self._init_df_and_parents(df)
        self.patchindex_to_cell = {v: k for k, v in self.cell_to_patchindex.items()}
        self.current_patches = list(range(len(self.grid.indices)))
        self.patch_neighbors = self._init_patch_nb()
        self.trans_probs = self._init_tp()
        self.prior_probs = self._init_uniform_pp() #dict
        self.parents = dict()
        self.predecessors = self._init_predecessors()
        self.entropy_dict = self._init_ed()
        self.adjacent_cells_loss = self._init_adjacent_cells_loss()
            
    def _init_df_and_parents(self, df):
        df['x_cell'] = df['x'].apply(self.grid.get_cell_index)
        df['y_cell'] = df['y'].apply(self.grid.get_cell_index)
        cell_to_index = {cell: idx for idx, cell in enumerate(self.grid.indices)}
        df['x_cell_id'] = df['x_cell'].map(cell_to_index)
        df['y_cell_id'] = df['y_cell'].map(cell_to_index)
        return df, cell_to_index
    
    def _init_patch_nb(self):
        cell_nb = self.grid.get_neighbors()
        patch_nb = {self.cell_to_patchindex[cell]: [self.cell_to_patchindex[n] for n in neighbors] for cell, neighbors in cell_nb.items()}
        return patch_nb
        
    def _init_tp(self):
        # Count transitions from (x_cell, c) → y_cell
        transition_counts = self.df.groupby(["x_cell_id", "c"])["y_cell_id"].value_counts().unstack(fill_value=0)
        # Normalize counts to get probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
        # Convert to nested dictionary: TP[s][a][s’] = probability #TODO: maybe this should be TP[s][s’][a]
        trans_probs_dict = {}
        for (s, a), series in transition_probs.stack().groupby(level=[0, 1]):
            trans_probs_dict.setdefault(int(s), {})[a] = {int(keys[2]): prob for keys, prob in series.items() if prob > 0}
        return trans_probs_dict 
    
    def _init_uniform_pp(self):
        uniform_prior_probs = {self.cell_to_patchindex[s]: 1/len(self.grid.indices) for s in self.grid.indices}
        return uniform_prior_probs

    def _init_predecessors(self):
        predecessors = dict()
        for s in self.trans_probs:
            for c in self.trans_probs[s]:
                for s_prime in self.trans_probs[s][c]:
                    if self.trans_probs[s][c][s_prime] > 0: #unnecessary, but for safety
                        predecessors.setdefault(s_prime, []).append((s, c))
                    else:
                        raise ValueError('There exists a Zero Probability in the Transition Probabilities Dict')
        return predecessors
    
    def _init_ed(self):
        entropy_dict = dict()
     
        for x in self.current_patches:
            combined_probs = self.get_prob_distr_over_all_actions(x)
            entropy_dict[x] = entropy(list(combined_probs.values()))
        
        return entropy_dict
    
    def get_prob_distr_over_all_actions(self, patch):
        combined_probs = dict()
        
        # Sum probabilities from all actions, weighting them equally
        for prob_dist in self.trans_probs[patch].values():
            for y, prob in prob_dist.items():
                combined_probs[y] =combined_probs.get(y, 0) + prob / len(self.trans_probs[patch].keys())  # Weighted sum
        return combined_probs

    
    def compute_merged_entropy(self, patch1,patch2, newpatch):
        #TODO test if correct!!!! maybe make more efficient
        merged_probs = dict()
        for patch in [patch1,patch2]:

            # Sum probabilities from all actions, weighting them equally
            combined_probs = self.get_prob_distr_over_all_actions(patch)

            for y, prob in combined_probs.items():
                if y != patch1 and y!= patch2:
                    merged_probs[y] = merged_probs.get(y,0) + self.prior_probs[patch]*prob
                else:
                    merged_probs[newpatch] = merged_probs.get(newpatch,0) + self.prior_probs[patch]*prob

        return entropy(list(merged_probs.values()))
    

    def calculate_entropy_loss(self, patch1, patch2):

        weight1, entropy1 = self.prior_probs[patch1], self.entropy_dict[patch1]
        weight2, entropy2 = self.prior_probs[patch2], self.entropy_dict[patch2]

        merged_entropy = self.compute_merged_entropy(patch1,patch2, 'z')

        entropy_loss = (weight1 + weight2) * merged_entropy - (weight1 * entropy1 + weight2 * entropy2)

        return entropy_loss


    def _init_adjacent_cells_loss(self):
        adjacent_cells_loss = dict()
        for cell, neighbors in self.patch_neighbors.items():
            for neighbor in neighbors:
                if neighbor > cell: #only calculate once for each pair
                    adjacent_cells_loss[(cell, neighbor)] = self.calculate_entropy_loss(cell, neighbor)
        return  adjacent_cells_loss 
        
    def count_patch_occurences_in_start(self):
        #This is the probability of the patch in the x_cell data
        # Count occurrences of (x_cell, c0)
        xcell_counts = self.df.groupby(["x_cell_id"]).size()
        # Normalize counts to get probabilities
        xcell_probs = xcell_counts.div(xcell_counts.sum(), level=0)
        # Convert to dictionary: PP[s] = probability
        xcell_probs_dict = xcell_probs.to_dict()
        return xcell_probs_dict
    