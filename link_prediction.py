import os
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from RGCN.custom_utils import load_relations, load_task_data_test
from RGCN.custom_models import RGCN
from collections import Counter
from itertools import combinations
import pandas as pd 
import networkx as nx

class LinkPrediction:
    def __init__(self, exp_num, gpu=-1):
        self.exp_num = exp_num
        self.gpu = gpu

        # Check for GPU availability
        self.use_cuda = self.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(self.gpu)

        # Load relations
        data_dir = './RGCN/data/gnn_dataset/block/'
        self.relation2id = load_relations(data_dir)
        self.num_relations = len(self.relation2id)

        # Initialize the model
        self.model = RGCN(num_relations=self.num_relations, num_bases=4, dropout=0.001)
        if self.use_cuda:
            self.model.cuda()

        # Load the saved model
        self.load_model()

    def create_graph(self, exp_num):
        input_folder = f"./experiments/exp_{exp_num}/2dsgg_data/"
        output_txt = f"./experiments/exp_{exp_num}/graph.txt"

        # Check if 2dsgg_data folder exists
        if os.path.exists(input_folder):
            unique_rows_counter = Counter()

            # Loop through all files in 2dsgg_data
            for file_name in os.listdir(input_folder):
                if file_name.startswith("") and file_name.endswith(".txt"):
                    file_path = os.path.join(input_folder, file_name)

                    # Read each file and collect unique rows based on the first three columns
                    with open(file_path, "r") as file:
                        for line in file:
                            # Modify the middle number if it is 1, 2, 3, or 4
                            parts = line.strip().split()
                            if len(parts) >= 3:  # Ensure there are at least 3 columns
                                if parts[1] == '1':
                                    parts[1] = '\\on'
                                elif parts[1] == '2':
                                    parts[1] = '\\next\\to'
                                elif parts[1] == '3':
                                    parts[1] = '\\in'
                                elif parts[1] == '4':
                                    parts[1] = '\\under'
                                # Check if parts[0] and parts[2] are the same
                                if parts[0] != parts[2]:
                                    # Use the first 3 columns as the unique key
                                    unique_key = tuple(parts[:3])
                                    unique_rows_counter[unique_key] += 1

            # Write unique rows and counts to graph.txt in descending order of count
            with open(output_txt, "w") as graph_file:
                for row, count in sorted(unique_rows_counter.items(), key=lambda x: x[1], reverse=True):
                    graph_file.write(f"{' '.join(row)} {count}\n")

            print(f"graph.txt has been created with {len(unique_rows_counter)} unique rows at {output_txt}")
        else:
            print(f"Folder '2dsgg_data' does not exist in {input_folder}")

    def create_samples(self, exp_num):
        input_txt = f"./experiments/exp_{exp_num}/entities.txt"
        output_txt = f"./experiments/exp_{exp_num}/pos_s.txt"

        try:
            # Read and parse the input file
            with open(input_txt, 'r') as infile:
                lines = infile.readlines()

            # Dictionary to track first-column values by second-column key
            column2_to_column1 = {}

            for line in lines:
                columns = line.strip().split(',')
                if len(columns) < 2:
                    continue  # Skip lines with insufficient columns
                
                col1 = columns[0].strip()
                col2 = columns[1].strip()
                
                if col2 not in column2_to_column1:
                    column2_to_column1[col2] = []
                column2_to_column1[col2].append(col1)
            
            # Write the results to the output file
            with open(output_txt, 'w') as outfile:
                for col2, col1_list in column2_to_column1.items():
                    if len(col1_list) > 1:  # Only process for duplicate second-column values
                        # Generate all pair combinations
                        for pair in combinations(col1_list, 2):
                            outfile.write(f"{pair[0]} {pair[1]}\n")

        except Exception as e:
            print(f"An error occurred: {e}")

    def load_model(self):
        model_path = './RGCN/checkpoints/best_mrr_model_270.pth'
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
    def predict(self, model, data, use_cuda): 
        model.eval()
        with torch.no_grad():
            device = torch.device('cuda' if use_cuda else 'cpu')
            data = data.to(device)
            
            entity_embeddings = data.entity_embeddings
            edge_index = data.edge_index
            edge_type = data.edge_type
            edge_norm = data.edge_norm
            samples = data.samples
            labels = data.labels

            if samples.shape[0] == 0:
                return 0.0  # No samples to evaluate

            # Forward pass to get entity embeddings
            entity_embedding = model(entity_embeddings, edge_index, edge_type, edge_norm)
            
            # Compute scores for the samples
            scores, _ = model.distmult(entity_embedding, samples)
            
            # To get probabilities
            predictions = torch.sigmoid(scores)
            
            # Convert probabilities to binary predictions (threshold at 0.5)
            predicted_labels = (predictions >= 0.5)

            print ("labels:", labels)
            
            return predicted_labels, samples

    def test(self, exp_num):

        test_dir = f'./experiments/exp_{exp_num}/'
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

        if os.path.isdir(test_dir):
            data = load_task_data_test(test_dir, self.relation2id)

        if data.samples.shape[0] == 0:
            warnings.warn(f"Warning: Dataset '{test_dir}' has no samples.", UserWarning)
            pred = None 
            sample = None
        else:
            pred, sample = self.predict(self.model, data, self.use_cuda)

        return pred, sample

    def run(self, exp_num):
        """
        1. Creates samples/graph (via self.create_samples / self.create_graph).
        2. Gets pred, sample from self.test(exp_num).
        3. Uses pred==True edges to form connected components.
        4. Merges entities in each connected component with weighted sums.
        5. Combines merged + unmerged entities into final_entities.
        6. Adjusts graph relationships and sums counts for identical edges.
        7. Returns final_entities, final_graph for further use.
        """

        # 1) Create data and get predictions
        self.create_samples(exp_num)
        self.create_graph(exp_num)
        pred, sample = self.test(exp_num)

        print("Original pred:", pred)
        print("Original sample:", sample)

        # ------------------------------------------------------------------------
        # You can override pred and sample for testing if you want:
        #
        # pred   = torch.tensor([False,  False,  True,  False])
        # sample = torch.tensor([[0, 1],
        #                        [0, 2],
        #                        [1, 2],
        #                        [4, 5]])
        # ------------------------------------------------------------------------

        # 2) Read entity information
        #    Format: id, label, x, y, z, s.v.t, e.v.t
        entities = pd.read_csv(f'./experiments/exp_{exp_num}/entities.txt', 
                            header=None, 
                            names=['id','label','x','y','z','s.v.t','e.v.t'])
        
        # Create a dictionary to map row index -> the actual entity ID
        # e.g. row 0 might have ID=100, row 1 might have ID=101, etc.
        row_to_id = dict(zip(entities.index, entities['id']))

        # 3) Read graph relationships
        #    Format: sub, rel, obj, count
        with open(f'./experiments/exp_{exp_num}/graph.txt', 'r') as f:
            graph_data = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    sub, rel, obj, count = parts
                    graph_data.append([int(sub), rel, int(obj), int(count)])
        graph_df = pd.DataFrame(graph_data, columns=['sub','rel','obj','count'])

        if pred is None and sample is None:
            final_entities = entities
            final_graph = graph_df

            # Print or return the final results
            print("Final Entities:")
            print(final_entities)
            print("\nFinal Graph:")
            print(final_graph)

        else:
            # 4) Convert pred/sample to standard Python list of edges.
            #    Edges that are 'True' => connect them in a graph.
            edges = []
            for i, is_true in enumerate(pred):
                if is_true:
                    # 'sample[i]' might be [row_index_sub, row_index_obj]
                    row_sub = int(sample[i][0])
                    row_obj = int(sample[i][1])
                    
                    # Map from row index to the entity's actual ID
                    sub_id = row_to_id[row_sub]
                    obj_id = row_to_id[row_obj]
                    edges.append((sub_id, obj_id))

            # 5) Create a NetworkX graph from these edges
            G = nx.Graph()
            G.add_edges_from(edges)

            # 6) Find connected components (i.e. sets of IDs to be merged)
            connected_comps = list(nx.connected_components(G))

            # Prepare results
            processed_ids = set()   # IDs that got merged
            merge_mapping = {}      # Map each ID -> primary/merged ID
            merged_entities = []    # Merged entity rows

            # 7) For each connected component, merge all its entities
            for comp in connected_comps:
                comp_ids = list(comp)  # e.g. [100, 101, 102] if those are the IDs
                subset = entities[entities['id'].isin(comp_ids)]

                # Mark these IDs as processed
                processed_ids.update(subset['id'].tolist())

                # Select a "primary" ID (lowest ID in this component)
                primary_id = subset['id'].min()

                # Build the mapping from each ID in the component to the primary ID
                for i in subset['id']:
                    merge_mapping[i] = primary_id

                # Weighted average for x,y,z by (e.v.t - s.v.t + 1)
                partition_weights = (subset['e.v.t'] - subset['s.v.t'] + 1).values
                total_weight = partition_weights.sum()

                weighted_x = (subset['x'] * partition_weights).sum() / total_weight
                weighted_y = (subset['y'] * partition_weights).sum() / total_weight
                weighted_z = (subset['z'] * partition_weights).sum() / total_weight

                # Combined time range
                min_svt = subset['s.v.t'].min()
                max_evt = subset['e.v.t'].max()

                # Retrieve a label for the merged entity
                label_primary = subset.loc[subset['id'] == primary_id, 'label'].values[0]

                # Build new merged row
                merged_entities.append({
                    'id':     primary_id,
                    'label':  label_primary,
                    'x':      weighted_x,
                    'y':      weighted_y,
                    'z':      weighted_z,
                    's.v.t':  min_svt,
                    'e.v.t':  max_evt
                })

            # 8) For entities not in any connected component, keep them as-is
            unchanged = entities[~entities['id'].isin(processed_ids)]
            merged_entities.extend(unchanged.to_dict(orient='records'))

            # Create final entities DataFrame
            final_entities = pd.DataFrame(merged_entities)

            # Remove rows where s.v.t == e.v.t (example condition you mentioned)
            final_entities = final_entities[final_entities['s.v.t'] != final_entities['e.v.t']]

            # 9) Merge relationships in graph_df
            #    Replace sub and obj with their primary IDs if they exist
            graph_df['sub'] = graph_df['sub'].replace(merge_mapping)
            graph_df['obj'] = graph_df['obj'].replace(merge_mapping)

            # If multiple edges become the same sub-rel-obj, sum their counts
            final_graph = graph_df.groupby(['sub','rel','obj'], as_index=False)['count'].sum()

            # Print or return the final results
            print("Final Entities:")
            print(final_entities)
            print("\nFinal Graph:")
            print(final_graph)

        return final_entities, final_graph
