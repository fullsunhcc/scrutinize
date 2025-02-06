import torch
import re
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Initialize the model and tokenizer
model_name = "./model/llama_model"  # Replace with your specific model name
max_seq_length = 512
dtype = torch.float16  # Adjust dtype if needed
load_in_4bit = True  # Enable efficient 4-bit loading for the model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable faster inference

# Define the prompt template
alpaca_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Ask the user to type the instruction and input
instruction = "This is robot task planning. Give me the steps."
wrong_result = ""#"Wrong_Result= \n"
Fail_Reason = ""#"Failed_Reason= blue_block_1 is not on yellow_block_1\n"
object_list = "Objects= ['police_car_1', 'ambulance_1', 'desk_1']\n"
command = "Command= Give me the step to place all objects on the desk."
task_input = wrong_result + Fail_Reason + object_list + command

# Tokenize the input prompt
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,  # User-provided instruction
            task_input,   # User-provided input details
            ""            # Leave response blank for generation
        )
    ],
    return_tensors="pt",
).to("cuda")



# Streamer for live token-by-token output generation
text_streamer = TextStreamer(tokenizer)

# Generate response
response = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)

# # Decode and print the response
decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)

print (decoded_response)

# Extract Action Steps and Object Relation using regular expressions
action_steps_pattern = r"Action Steps=\n(.*?)\nObject Relation="
object_relation_pattern = r"Object Relation=\n(.*?)$"

# Extract Action Steps
action_steps_match = re.search(action_steps_pattern, decoded_response, re.DOTALL)
action_steps = action_steps_match.group(1).strip() if action_steps_match else "Not found"

# Extract Object Relation
object_relation_match = re.search(object_relation_pattern, decoded_response, re.DOTALL)
object_relation = object_relation_match.group(1).strip() if object_relation_match else "Not found"

# Print the extracted parts
print("\nAction Steps:")
print(action_steps)

action_steps_lines = action_steps.splitlines()

# Print each step on a separate line
print("\nAction Steps (separate lines):")
for step in action_steps_lines:
    print(step)

print("\nObject Relation:")
print(object_relation)