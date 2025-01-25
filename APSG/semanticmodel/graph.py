import ipdb
with open('/mnt/yangzhenyu/SemanticFlowGraph-main/SemanticFlowGraph/src/0.txt', 'r') as file:
    sour_lines = file.readlines()
with open('/mnt/yangzhenyu/SemanticFlowGraph-main/SemanticFlowGraph/src/result0.txt', 'r') as file:
    res_lines = file.readlines() 
newfile = open('/mnt/yangzhenyu/SemanticFlowGraph-main/SemanticFlowGraph/src/new_result0.txt', mode='w', encoding='utf-8')
tag=1
sent= []
#ipdb.set_trace()
for i in range(len(sour_lines)):
    # #ipdb.set_trace()
    word = []
    sourp = 0
    for j in range(tag,len(res_lines),1):
        sour_line = sour_lines[i].strip()
        res_line = res_lines[j].strip()
        if res_line == '?':
            continue
        #    #ipdb.set_trace()
        if res_line == 'Graph Nodes and Related Information:':
            tag = j
            # print(word)
            if bool(word):
                sent.append(word)
            break 
        index = sour_line[sourp:].find(res_line)
        if index !=-1:
            sourp=sourp+index+len(res_line)
            word.append(res_line)
        else:
            tag = j
            print("一句话结束了")   
            sent.append(word)
            break
#ipdb.set_trace()
newfile.write('Sentence Sequence for code:\n')

for i in sent:
    newfile.write(' '.join(i)+'\n')


sentnum=[]
for i in range(len(sent)):
    for j in range(len(sent[i])):
        sentnum.append(i)
#ipdb.set_trace()
senttag=[1 for _ in range(len(sent))]
num = nums = 0
sentindex1=[]
sentindex=[]

newfile.write('Graph Nodes and Related Information:\n')
infor = tag
for i in range(tag+1,len(res_lines),1):
   
    node = res_lines[i].strip()
    if node == 'Graph Edges:':
            tag = i
            break 
    _,att = node.split(":")
    att = att.strip().split(",")

    tok_index = att[0].split("=")[1]
    if tok_index == "---":
        # #ipdb.set_trace()
        num = num+1
        newfile.write("Sent "+str(num)+" : "+",".join(att)+'\n')
        sentindex1.append(len(sentindex1)+len(sentindex))
    else:
        #ipdb.set_trace()
        print(node)
        tok_index =eval(tok_index)
        if senttag[sentnum[tok_index-1]]:
            num =num +1
            nums = nums +1
            newfile.write("Sent "+str(num)+" : "+" ".join(sent[sentnum[tok_index-1]])+'\n')
            senttag[sentnum[tok_index-1]]=0
        sentindex.append(nums-1)
max =0
#ipdb.set_trace()
 

for i in sentindex1:
    a=1
    for j in range(i,len(sentindex),1):
        if j!=0 and sentindex[j]==sentindex[j-1] and a:
            continue
        a=0
        sentindex[j] =sentindex[j]+1
    if i ==0:
        sentindex.insert(0,0)
    else:
        sentindex.insert(i,sentindex[i-1]+1)
#ipdb.set_trace()
newfile.write('Graph Edges:\n')

linel=[]
sameline= [[] for _ in range(sentindex[-1]+1)]
for i in range(tag+1,len(res_lines),1):
    line = res_lines[i].strip()
    start, end = line.split("->")
    start = eval(start)
    end = eval(end)

    if sentindex[start-1]!=sentindex[end-1]:
        if [sentindex[start-1]+1,sentindex[end-1]+1] in linel:
            continue
        else:
            newfile.write(str(sentindex[start-1]+1)+"->"+str(sentindex[end-1]+1)+'\n')
            linel.append([sentindex[start-1]+1,sentindex[end-1]+1])
