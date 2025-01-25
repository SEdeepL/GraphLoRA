import sys
import os
import networkx as nx
def distancetopatch(graph_path):
    entries = os.listdir(graph_path)
    subdirectories = [d for d in entries if os.path.isdir(os.path.join(graph_path, d))]
    for i in subdirectories:
        path=[]
        patch=[]
        context=[]
        directory=graph_path+"/"+i
        graph = open(directory, 'r')
        graph_lines = graph.readlines()
        for index, value in enumerate(graph_lines):
            if value == "Graph Edges:":
                path_index=index
        for j in range(path_index+1,len(graph_lines)):
            path.append(graph_lines[j])
        G = nx.Graph()
        G_list=[]
        for j in range(len(path)):
            start,end = path[i].split("->")
            G_list.append((start,end))
        G.add_edges_from()
        for index, value in enumerate(graph_lines):
            if value == "Graph Nodes and Related Information:":
                nodes_index=index
        for j in range(nodes_index+1,len(graph_lines)):
            if graph_lines[j].find("patch"):
                patch.append(j)
            if graph_lines[j].find("context"):
                context.append(j)
        for k in context:
            min =100
            for p in patch:
                shortest_path = nx.shortest_path_length(G, source=context[k], target=patch[p])
                if min >shortest_path:
                    min=shortest_path
            for l in range(nodes_index+1,len(graph_lines)):
                if graph_lines[l].find(str(k)):
                    graph_lines[l] = graph_lines[l]+"distancetopatch"+str(min)
if __name__ == "__main__":
    # 获取传入的参数
    graph = sys.argv[1]
    distancetopatch(graph)
