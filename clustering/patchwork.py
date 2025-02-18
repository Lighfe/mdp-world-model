import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)



class Patchwork:
    

    def __init__(self, grid, df, configs_dict):
        #TODO save patches also somehow geometrically?
        self.grid = grid
        self.configs_dict = configs_dict #necessary?
        self.df, self.cell_to_index = self._init_df_and_parents(df)
        self.current_patches = list(range(len(self.grid.indices)))
        self.trans_probs = self._init_tp()
        self.prior_probs = self._init_pp()
        self.parents = dict()
        self.predecessors = self._init_predecessors()
            
    def _init_df_and_parents(self, df):
        df['x_cell'] = df['x'].apply(self.grid.get_cell_index)
        df['y_cell'] = df['y'].apply(self.grid.get_cell_index)
        cell_to_index = {cell: idx for idx, cell in enumerate(self.grid.indices)}
        df['x_cell_id'] = df['x_cell'].map(cell_to_index)
        df['y_cell_id'] = df['y_cell'].map(cell_to_index)
        return df, cell_to_index
        
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
    
    def _init_pp(self):
        #TODO: This is not the prior probability, but the probability of the patch in the data
        # Count occurrences of (x_cell, c0)
        prior_counts = self.df.groupby(["x_cell_id"]).size()
        # Normalize counts to get probabilities
        prior_probs = prior_counts.div(prior_counts.sum(), level=0)
        # Convert to dictionary: PP[s] = probability
        prior_probs_dict = prior_probs.to_dict()
        return prior_probs_dict
    
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