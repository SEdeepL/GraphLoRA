import torch
from transformers import TrainingArguments,LlamaTokenizer, AutoModelForSequenceClassification,AutoTokenizer
from peft import PeftModel,LoraConfig
import ipdb
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
# ipdb.set_trace()
base_model_path = './llama2-hf'
finetune_model_path ='/home/sdu/llama-main/check_point/checkpoint-200'
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(base_model_path, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16, num_labels=2)
model = PeftModel.from_pretrained(model, finetune_model_path)
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
test_dataset=[]
tf = open("./data/test.txt") 
graph_test="./data/graph_test"
entries = os.listdir(graph_test)
tfgraph = [d for d in entries if os.path.isdir(os.path.join(graph_train, d))]
line = tf.readlines()
for i in range(len(line)):
    patch,result,_=line[i].split("	")
    label = eval(result)
    train_graph = tfgraph[i]
    nodes, edges,attris =  get_graph(graph_test+tfgraph[i])
    nodeid = []
    for i in range(len(nodes)):
        nodeid.append(tokenizer(nodes[i])["input_ids"])
    edgesid = torch.tensor(edges)
    attrisid = torch.tensor(attris)
    test_dataset.append({'text': '<s>'+'Assess whether the code is correct:\nInput:'+ patch + '</s>',"nodes":nodeid,"edges":edgesid,"attri":attrisid,"labels":label})
    line = tf.readline()
tf.close()

test_dataset = Dataset.from_dict({key: [dic[key] for dic in test_dataset] for key in test_dataset[0]})
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)

def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)  # 获取预测标签
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
training_aruments = TrainingArguments(
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
peft_config = LoraConfig(
    r=8,
    target_modules=['q_proj','v_proj'],
    lora_dropout = 0.05,
    bias='none',
    task_type='CAUSAL_LM'
)
trainer = SFTTrainer(
    model=model,
    eval_dataset=encoded_test_dataset,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_aruments
)
model.eval()
results = trainer.evaluate()

# 打印评估结果
print(f"Accuracy: {results['eval_accuracy']}")
print(f"Precision: {results['eval_precision']}")
print(f"Recall: {results['eval_recall']}")
print(f"F1: {results['eval_f1']}")
