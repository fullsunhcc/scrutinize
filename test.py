import pandas as pd
from collections import defaultdict

def process_entities(pred, sample, entities, exp_num):
    if len(pred) > 0:
        # Find connected components
        connected_components = defaultdict(set)
        for i, is_true in enumerate(pred):
            if is_true:
                u, v = sample[i]
                connected_components[u].add(u)
                connected_components[u].add(v)
                connected_components[v].add(u)
                connected_components[v].add(v)

        # Merge groups into clusters
        def find_connected_clusters(components):
            visited = set()
            clusters = []

            def dfs(node, cluster):
                if node in visited:
                    return
                visited.add(node)
                cluster.append(node)
                for neighbor in components[node]:
                    dfs(neighbor, cluster)

            for node in components:
                if node not in visited:
                    cluster = []
                    dfs(node, cluster)
                    clusters.append(cluster)
            return clusters

        clusters = find_connected_clusters(connected_components)

        # Process clusters to merge entities
        result = []
        for cluster in clusters:
            selected_rows = entities[entities['id'].isin(cluster)]
            primary_id = selected_rows['id'].min()

            partition_weights = (selected_rows['e.v.t'] - selected_rows['s.v.t']).values
            total_weight = partition_weights.sum()
            weighted_x = (selected_rows['x'] * partition_weights).sum() / total_weight
            weighted_y = (selected_rows['y'] * partition_weights).sum() / total_weight
            weighted_z = (selected_rows['z'] * partition_weights).sum() / total_weight

            min_svt = selected_rows['s.v.t'].min()
            max_evt = selected_rows['e.v.t'].max()

            new_row = {
                'id': primary_id,
                'label': selected_rows['label'].iloc[0],
                'x': weighted_x,
                'y': weighted_y,
                'z': weighted_z,
                's.v.t': min_svt,
                'e.v.t': max_evt
            }
            result.append(new_row)

        # Add unconnected entities as-is
        processed_ids = set(row['id'] for row in result)
        for i, is_true in enumerate(pred):
            if not is_true:
                row = entities.iloc[int(sample[i][0])].to_dict()
                if row['id'] not in processed_ids:
                    result.append(row)

        return result

# Example Usage
if __name__ == "__main__":
    # Example inputs
    pred = [True, True, True, False]
    sample = [
        [0, 1],
        [0, 2],
        [1, 2],
        [3, 5]
    ]

    # Entities data as a pandas DataFrame
    entities_data = [
        {"id": 0, "label": "A", "x": 1.0, "y": 2.0, "z": 3.0, "s.v.t": 10, "e.v.t": 20},
        {"id": 1, "label": "A", "x": 2.0, "y": 3.0, "z": 4.0, "s.v.t": 15, "e.v.t": 25},
        {"id": 2, "label": "A", "x": 3.0, "y": 4.0, "z": 5.0, "s.v.t": 20, "e.v.t": 30},
        {"id": 3, "label": "B", "x": 5.0, "y": 6.0, "z": 7.0, "s.v.t": 10, "e.v.t": 15},
        {"id": 5, "label": "B", "x": 6.0, "y": 7.0, "z": 8.0, "s.v.t": 20, "e.v.t": 25},
    ]
    entities = pd.DataFrame(entities_data)

    exp_num = 1
    result = process_entities(pred, sample, entities, exp_num)

    # Print the results
    print("Processed Results:")
    for row in result:
        print(row)
