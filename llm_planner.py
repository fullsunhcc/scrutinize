import torch
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer

class LLMPlanner:
    """
    LLM Robot Planner
    """
    def __init__(self, exp_num):
        self.exp_num = exp_num

    def task_planning(self, instruction, wrong_result, fail_reason, task_input, obj_list):
        # Initialize the model and tokenizer
        llm_model = "./model/llama_model"  # Replace with your specific model name
        max_seq_length = 512
        dtype = torch.float16
        load_in_4bit = True

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_model,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        # Prompt template
        alpaca_prompt = """\n### Instruction:\n{}\n### Input:\n{}\n### Output:\n"""

        # Input handling
        llm_input = wrong_result + fail_reason + obj_list + task_input

        instruction = instruction

        # Tokenize input
        inputs = tokenizer(
            [alpaca_prompt.format(instruction, llm_input, "")],
            return_tensors="pt",
        ).to("cuda")

        # Streamer for live output
        text_streamer = TextStreamer(tokenizer)

        # Generate response
        response = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)

        # Decode and print the response
        decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)

        # Extract Action Steps and Object Relation
        action_steps_pattern = r"(?:Action Steps=|Action Step=)\n(.*?)\nObject Relation="
        object_relation_pattern = r"Object Relation=\n(.*?)$"

        action_steps_match = re.search(action_steps_pattern, decoded_response, re.DOTALL)
        object_relation_match = re.search(object_relation_pattern, decoded_response, re.DOTALL)

        action_steps = action_steps_match.group(1).strip().splitlines() if action_steps_match else ["No action steps found"]
        object_relation = object_relation_match.group(1).strip().splitlines() if object_relation_match else ["No object relation found"]

        return action_steps, object_relation
        
    def extract_(self, data):
        extracted_data = []
        for step in data:
            # Extract the method name
            start_index = step.find('.') + 1
            end_index = step.find('(')
            
            if start_index != -1 and end_index != -1:
                method_name = step[start_index:end_index].strip()
                
                # Remove known prefixes like "robot." or "object."
                for prefix in ["robot.", "object."]:
                    if method_name.startswith(prefix):
                        method_name = method_name[len(prefix):]
                
                # Extract the arguments inside the parentheses
                args_start = step.find('(')
                args_end = step.find(')')
                if args_start != -1 and args_end != -1:
                    arguments = step[args_start + 1:args_end].strip()  # Remove parentheses
                    args_list = [arg.strip().strip("'") for arg in arguments.split(',')]  # Split and clean arguments
                    if len(args_list) == 1:
                        # Single argument case
                        arguments = {'object_a': args_list[0]}
                    elif len(args_list) == 2:
                        # Two arguments case
                        arguments = {'object_a': args_list[0], 'object_b': args_list[1]}
                    else:
                        # Unexpected argument format
                        arguments = None
                else:
                    arguments = None  # No arguments found
                
                # Combine method name and arguments
                extracted_data.append({'method': method_name, 'arguments': arguments})
        
        return extracted_data

