import torch
from peft import LoraConfig, get_peft_model 
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import ipdb
train_dataset=[]
val_dataset=[]

tf = open("./data/train.txt") 
graph_train="./data/graph_train"
entries = os.listdir(graph_train)
tfgraph = [d for d in entries if os.path.isdir(os.path.join(graph_train, d))]
line = tf.readline()
for i in range(len(line)):
    patch,result,_=line[i].split("	")
    train_graph = tfgraph[i]
    graph =  open(graph_train+tfgraph[i])
    graph_line = graph.readline()
    train_dataset.append({'text': 'Assess whether the code is correct:\nInput:'+ patch + '\nOutput:'+ result + '</s>',"graph":graph_line})
    line = tf.readline()
tf.close()
ef = open("./data/eval.txt") 
graph_eval="./data/graph_eval"
entries = os.listdir(graph_train)
efgraph = [d for d in entries if os.path.isdir(os.path.join(graph_eval, d))]
line = ef.readline()
for i in range(len(line)):
    patch,result,_=line.split("	")
    val_graph = efgraph[i]
    graph =  open(graph_train+efgraph[i])
    graph_line = graph.readline()
    val_dataset.append({'text': 'Assess whether the code is correct:\nInput:'+ patch + '\nOutput:'+ result +'</s>',"graph":eval_graph})
    line = ef.readline()
ef.close()
train_dataset = Dataset.from_dict({key: [dic[key] for dic in train_dataset] for key in train_dataset[0]})
val_dataset = Dataset.from_dict({key: [dic[key] for dic in val_dataset] for key in val_dataset[0]})
# ipdb.set_trace()
peft_config = LoraConfig(
    r=8,
    target_modules=['q_proj','v_proj'],
    lora_dropout = 0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
# 配置训练参数
training_aruments = TrainingArguments(
    output_dir="./check_point",
    per_device_train_batch_size=64,
    optim='adamw_torch',
    learning_rate=10e-4,
    eval_steps=50,
    save_steps=100,
    logging_steps=20,
    evaluation_strategy='steps',
    group_by_length=False,
    # num_train_epochs=2，
    max_steps=200,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=100
)
model_name ='./llama2-hf/'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
ipdb.set_trace()

# model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'
# ipdb.set_trace()
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_aruments
)
trainer.train()
trainer.model.save_pretrained("./trainedmodel")
