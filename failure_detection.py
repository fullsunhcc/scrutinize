import pandas as pd 
import numpy as np
import link_prediction as LinkPrediction 

class FailureDetection:

    def get_object_list(self, entities, graph):
        """
        Parameters:
            entities (pd.DataFrame): columns = [id, label, x, y, z, start_valid_time, end_valid_time]
            graph (pd.DataFrame):    columns = [sub, rel, obj, count]
                                    (where 'sub' and 'obj' are numeric labels)

        Returns:
            tuple: (object_list, summary_string)
                - object_list: list of dicts
                - summary_string: string summary like "Objects= ['desk_1', 'desk_2', ...]"
        """

        # Mapping numeric labels to string names
        label_mapping = {
            0: 'red_block',
            1: 'yellow_block',
            2: 'blue_block',
            3: 'red_plate',
            4: 'yellow_plate',
            5: 'blue_plate',
            6: 'desk',
            7: 'police_car',
            8: 'ambulance',
            9: 'pot',
            10: 'carrot',
            11: 'daikon',
            12: 'cucumber',
            13: 'microwave',
            14: 'banana',
            15: 'kiwi'
        }

        # Calculate counts for sub and obj labels
        sub_counts = graph.groupby('sub')['count'].sum().to_dict()
        obj_counts = graph.groupby('obj')['count'].sum().to_dict()

        # Merge counts into id_counts
        id_counts = {}
        for label_id in label_mapping.keys():
            sub_count = sub_counts.get(label_id, 0)
            obj_count = obj_counts.get(label_id, 0)
            id_counts[label_id] = sub_count + obj_count

        # print (id_counts)

        # Initialize label numbering
        label_start_number = {}
        for label in id_counts:
            label_start_number[label] = 1

        # Sort labels in entities based on id_counts
        entities = entities.sort_values(by='id', key=lambda x: x.map(id_counts), ascending=False)

        # print (entities)

        object_list = []
        object_names = []

        for _, row in entities.iterrows():
            numeric_label = row['label']
            label_str = label_mapping.get(numeric_label, f"unknown_{numeric_label}")

            # Use the start number based on label_start_number
            unique_object_name = f"{label_str}_{label_start_number[numeric_label]}"
            label_start_number[numeric_label] += 1

            object_names.append(unique_object_name)

            transformed_object = {
                'label': numeric_label,
                'object': unique_object_name,
                'x': row['x'],
                'y': row['y'],
                'z': row['z']
            }
            object_list.append(transformed_object)

        summary_string = f"Objects= {object_names}\n"
        return object_list, summary_string

    @staticmethod
    def matching_id(entities, object_list):
        '''
        Match entities with object_list based on the smallest Euclidean distance.

        Parameters:
        entities (pd.DataFrame): [id, label, x, y, z, start_valid_time, end_valid_time]
        object_list (list): [{'label': , 'object': 'red_block_1', 'x': 0.44, 'y': 0.132, 'z': 0.09}, ...]

        Returns:
        tuple: updated entities with matched object information and a list of matches [id, object]
        '''

        matches = []
        if object_list == None:
            print ("error")
        
        else: 
            # List to store the final [id, object] mappings     
            # Iterate over unique labels in the object list
            for label in {obj['label'] for obj in object_list}:
                # Filter entities and objects by the current label
                label_entities = entities[entities['label'] == label]
                label_objects = [obj for obj in object_list if obj['label'] == label]
                
                if len(label_entities) == 1 and len(label_objects) == 1:
                    # If there's only one entity and one object with this label, match them directly
                    matches.append([label_entities.iloc[0]['id'], label_objects[0]['object']])
                else:
                    # If there are multiple entities or objects, calculate distances and find the best match
                    distances = []
                    for _, entity in label_entities.iterrows():
                        for obj in label_objects:
                            dist = np.sqrt(
                                (entity['x'] - obj['x'])**2 +
                                (entity['y'] - obj['y'])**2 +
                                (entity['z'] - obj['z'])**2
                            )
                            distances.append((dist, entity['id'], obj['object']))
                    
                    # Sort by distance and select the best matches
                    distances.sort()  # Sort by the first element (distance)
                    selected_objects = set()
                    for dist, entity_id, obj_name in distances:
                        if obj_name not in selected_objects:
                            matches.append([entity_id, obj_name])
                            selected_objects.add(obj_name)
                            # Break once all objects for this label are matched
                            if len(selected_objects) == len(label_objects):
                                break

        return matches

    @staticmethod    
    def graph_to_SGG(matches, graph):
        """
        Convert a graph to a Scene Graph Generation (SGG) format, and replace object and subject names with their corresponding IDs after ensuring bidirectional relationships.

        Parameters:
        matches (list): List of [id, object] pairs representing the mapping between IDs and objects.
        graph (pd.DataFrame): Input graph DataFrame with columns ['sub', 'rel', 'obj', 'count'].

        Returns:
        list: Scene Graph Generation (SGG) list with IDs replacing object and subject names.
        """

        # Filter the graph where the count is greater than or equal to 30
        filtered_graph = graph[graph["count"] >= 30].copy()

        # Replace "\on" with 1 and "\next\to" with 2
        filtered_graph.loc[:, "rel"] = filtered_graph["rel"].replace({"\\on": 1, "\\next\\to": 2})

        # Convert the filtered graph to a list of tuples
        SGG = filtered_graph[["sub", "rel", "obj"]].values.tolist()

        # Ensure bidirectional relationships for rel == 2
        unique_set = set(map(tuple, SGG))  # Use a set for efficient lookup
        for sub, rel, obj in list(SGG):  # Duplicate list for safe modification
            if rel == 2 and (obj, rel, sub) not in unique_set:
                SGG.append([obj, rel, sub])
                unique_set.add((obj, rel, sub))

        # Create a mapping dictionary from matches
        object_to_id = {int(float(id_)): obj for id_, obj in matches}

        # print (object_to_id)

        # Replace sub and obj names with their corresponding IDs using the matches
        final_SGG = []
        for sub, rel, obj in SGG:
            sub_id = object_to_id.get(sub)
            obj_id = object_to_id.get(obj)
            if sub_id is not None and obj_id is not None:
                final_SGG.append([sub_id, rel, obj_id])

        return final_SGG
    
    @staticmethod
    def relations_to_GTSGG(relations, i):
        """
        Convert a list of relations into a structured GTSGG format using object labels.

        Parameters:

            object_list (list): [{'label': , 'object': 'red_block_1', 'x': 0.44, 'y': 0.132, 'z': 0.09}, ...]
            i (int): The number of relations to process.

        Returns:
            GTSGG (list): A list of [sub_label, rel, obj_label] entries representing relationships.
        """
        # Limit relations to the first i elements
        relations = relations[:i+1]

        # Map method names to relation IDs
        method_to_relation = {
            'relation_on': 1,
            'relation_next_to': 2,
            'relation_in': 3,
            'relation_under': 4
        }

        GTSGG = []  # Initialize the GTSGG list

        # Process each relation in the truncated list
        for item in relations:
            # Extract method and arguments safely
            method = item.get('method')
            arguments = item.get('arguments', {})

            # Get the corresponding relation ID
            rel = method_to_relation.get(method)

            if rel is not None:
                # Extract subject (sub) and object (obj)
                sub = arguments.get('object_a')
                obj = arguments.get('object_b')

                GTSGG.append([sub, rel, obj])

        return GTSGG

    def failure_detection(self, relations, object_list, entities, graph, i, Task_GTSGG):
        # Match entities with the object list
        matches = FailureDetection.matching_id(entities, object_list)

        # Convert the graph to Scene Graph Generator (SGG) format
        SGG = FailureDetection.graph_to_SGG(matches, graph)

        print ("SGG: ", SGG)

        if i == len(relations) - 1:
            GTSGG = Task_GTSGG
        else:
            # Convert relations to Ground Truth Scene Graph Generator (GTSGG) format
            GTSGG = FailureDetection.relations_to_GTSGG(relations, i)

        print ("GTSGG: ", GTSGG)

        relation_mapping ={1: 'on', 2: 'next to', 3: 'in', 4: 'under'}

        # Initialize lists to store mismatches
        wrong = [item for item in SGG if item not in GTSGG]  # Items in SGG but not in GTSGG
        fail = [item for item in GTSGG if item not in SGG]  # Items in GTSGG but not in SGG

        # Initialize result strings
        wrong_result = ""
        fail_reason = ""

        # Determine task success based on the presence of mismatches
        if not fail:
            task_success = True
        else:
            task_success = False

            # Build the wrong result string
            wrong_result = "".join(f"{w[0]} is {relation_mapping[w[1]]} {w[2]}, " for w in wrong)
            # Remove the trailing ", " if the string is not empty
            if wrong_result:
                wrong_result ="Wrong_Result= " + wrong_result[:-2] + "\n"
            else:
                wrong_result ="Wrong_Result= \n"

            # Build the fail reason string
            fail_reason = "".join(f"{f[0]} is not {relation_mapping[f[1]]} {f[2]}, " for f in fail)
            # Remove the trailing ", " if the string is not empty
            if fail_reason:
                fail_reason = "Failed_Reason= "+ fail_reason[:-2] + "\n"
            else:
                fail_reason ="Failed_Reason= \n"

        return task_success, wrong_result, fail_reason


