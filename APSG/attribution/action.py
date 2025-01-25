import sys
import os
def action(directory_path,graph_path):
    num =0 
    entries = os.listdir(directory_path)
    subdirectories = [d for d in entries if os.path.isdir(os.path.join(directory_path, d))]
    for i in subdirectories:
        if i == "Many":
            continue
        directory1=directory_path+"/"+i
        entries1 = os.listdir(directory1)
        for j in entries1:
            directory2 =  directory1+"/"+j
            entries2 = os.listdir(directory2)
            for k in entries2:

                directory3 =  directory2+"/"+k
                print(directory3)
                with open(directory3, 'r') as file:
                    lines = file.readlines()
                bug=[]
                patch=[]
                bugtext = []
                patchtext = []
                add = False
                delete = False
                for line in lines:
                    line = line.strip()
                    if line[:3]=="---" or line[:3]=="+++" or line == "" or line[:4]=="diff" or line[0]=="/":
                        continue
                    elif line[:2] == "@@":
                        print("start----------------")
                        if len(bugtext)!=0 and len(patchtext)!=0:
                            # ipdb.set_trace()
                            bug.append(bugtext.copy())
                            patch.append(patchtext.copy())
                            bugtext.clear()
                            patchtext.clear()
                    elif line[0] == "-":
                        delete=True
                    elif line[0] == "+":
                        add = True
                ac = ""
                if delete & add:
                    ac="replace"
                elif delete:
                    ac="delete"
                elif add:
                    ac="add"
                graph = open('graph_path'+str(num)+".txt", 'r')
                lines = graph.readlines()
                for i in line:
                    if i.find("patch"):
                        i= i+"action:"+ac
if __name__ == "__main__":
    # 获取传入的参数
    source_file = sys.argv[1]
    graph = sys.argv[2]
    action(source_file,graph)
