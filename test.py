import torch
from transformers import LlamaTokenizer, LlamaForCausalLM,AutoTokenizer
from peft import PeftModel

base_model_path = './llama2-hf'
finetune_model_path ='./llama-main/check_point'
nerged_model_path='./llama-main/llama-2-7b-merged'
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(base_model_path, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, finetune_model_path)
sf = open("./data/overfitting/small/Time_test_v1.csv")
line = sf.readline()
while line:
    label,code=line.split("~")
    test_prompt = 'Assess whether the code is correct:\nInput:'+ code + '\nOutput:'

    model_input = tokenizer(test_prompt, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        res = model.generate(**model_input, max_new_tokens=100)[0]
        print(tokenizer.decode(res,skip_special_tokens=True))
