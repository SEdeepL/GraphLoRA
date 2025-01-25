def linearization(graph):    
    nodes_list = []
    for node in graph["nodes"]:
        nodes_list.append([node["id"], node["content"], node["editdistance"], node["entropyscore "], node["repairaction"], node["antipattern"], node["distance"], node["statement"], node["operator"], node["controltype"], node["nestedcontrol"], node["variabletype"], node["variablerole"]])

    edges_list = [] 
    for u, v in graph["edges"]:
        edges_list.append([u, v])


    graph_linearized = nodes_list + edges_list

    return (graph_linearized)
