import sys
import os
def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j  


    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]
            else:

                dp[i][j] = 1 + min(dp[i - 1][j],    
                                   dp[i][j - 1],   
                                   dp[i - 1][j - 1])  

    return dp[m][n]

if __name__ == "__main__":
    patch_file = sys.argv[1]
    bug_file = sys.argv[2]
    graph_file = sys.argv[3]
    patch = open(patch_file, 'r')
    patch_lines = patch.readlines()
    bug = open(bug_file, 'r')
    bug_lines = bug.readlines()
    entries = os.listdir(graph_file)
    subdirectories = [d for d in entries if os.path.isdir(os.path.join(graph_file, d))]
    for i in range(len(patch_lines)):
        distance = edit_distance(patch_lines[i], bug_lines[i])
        graph = open(graph_file+subdirectories[i], 'r')
        for index, value in enumerate(graph):
            if value == "Graph Nodes and Related Information:":
                nodes_index=index
        for j in range(nodes_index,len(graph)):
            if graph[j].find("patch"):
                graph[j] =graph[j]+"editdistanc"+str(distance)
