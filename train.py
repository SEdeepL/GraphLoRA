import torch
from peft import LoraConfig, get_peft_model 
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import ipdb
train_dataset=[]
val_dataset=[]
def get_graph(path):
    graph= open(path)
    graph_line = graph.readlines()
    nodep = graph_line.index("Graph Nodes and Related Information:\n")
    edgep = graph_line.index("Graph Edges:\n")
    nodes=[]
    edges=[]
    attris=[]
    for i in range(nodep+1,edgep):
        star = graph_line[i].index("[")
        end = graph_line[i].index("]")
        n = graph_line[i][star+1:end]
        cont = n.split(",")
        _,index = cont[0].split("=")
        if index == "---":
            nodes.append(cont[1].split("=")[1])
        else:
            nodes.append(graph_line[index].strip())
        attri=[]
        for k in range(1,len(cont)):
            attri.append(cont[i].split("=")[1])
        attri.append(attri)
    heads=[]
    tails=[]
    for j in range(edgep+1,len(graph_line)):
        e = graph_line[j].strip()
        edgelist = e.split(" -> ")
        head = eval(edgelist[0])
        tail = eval(edgelist[1])
        heads.append(head)
        tails.append(tail)
    edges.append(heads)
    edges.append(tails)
    return nodes, edges,attris
tf = open("./data/train.txt") 
graph_train="./data/graph_train"
entries = os.listdir(graph_train)
tfgraph = [d for d in entries if os.path.isdir(os.path.join(graph_train, d))]
line = tf.readline()
for i in range(len(line)):
    patch,result,_=line[i].split("	")
    label = eval(result)
    train_graph = tfgraph[i]
    nodes, edges,attris =  get_graph(graph_train+tfgraph[i])
    train_dataset.append({'text': '<s>'+'Assess whether the code is correct:\nInput:'+ patch + '\nOutput:'+ result + '</s>',"nodes":nodes,"edges":edges,"attri":attris,"labels":label})
    line = tf.readline()
tf.close()
ef = open("./data/eval.txt") 
graph_eval="./data/graph_eval"
entries = os.listdir(graph_train)
efgraph = [d for d in entries if os.path.isdir(os.path.join(graph_eval, d))]
line = ef.readline()
for i in range(len(line)):
    patch,result,_=line.split("	")
    label = eval(result)
    val_graph = efgraph[i]
    nodes, edges,attris =  get_graph(graph_train+efgraph[i])
    val_dataset.append({'text': '<s>'+'Assess whether the code is correct:\nInput:'+ patch + '\nOutput:'+ result +'</s>',"nodes":nodes,"edges":edges,"attri":attris,"labels":label})
    line = ef.readline()
ef.close()
train_dataset = Dataset.from_dict({key: [dic[key] for dic in train_dataset] for key in train_dataset[0]})
val_dataset = Dataset.from_dict({key: [dic[key] for dic in val_dataset] for key in val_dataset[0]})
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'nodes','edges','attris','labels'])
encoded_eval_dataset = val_dataset.map(preprocess_function, batched=True)
encoded_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'nodes','edges','attris','labels'])
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
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_eval_dataset,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_aruments
)
trainer.train()
trainer.model.save_pretrained("./trainedmodel")
