import re
import os
import sys

def check_java_statements(java_code):
    # 赋值语句 (赋值运算符)
    assignment_pattern = r"\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[\+\-\*/\%\^&|\=]{1,2}.*\s*;"
    
    # try-catch 语句
    try_catch_pattern = r"\btry\b.*\bcatch\b"
    
    # 方法调用语句 (简单检测方法调用形式：对象.方法() 或 类.方法())
    method_call_pattern = r"\w+\.\w+\(.*\)\s*;"
    
    # 返回语句
    return_pattern = r"\breturn\b\s*.*\s*;"
    
    # 检查是否包含赋值语句
    has_assignment = bool(re.search(assignment_pattern, java_code))
    
    # 检查是否包含 try-catch 语句
    has_try_catch = bool(re.search(try_catch_pattern, java_code, re.DOTALL))
    
    # 检查是否包含方法调用语句
    has_method_call = bool(re.search(method_call_pattern, java_code))
    
    # 检查是否包含返回语句
    has_return_statement = bool(re.search(return_pattern, java_code))
    
    # 返回检查结果
    return {
        "Has Assignment": has_assignment,
        "Has Try-Catch": has_try_catch,
        "Has Method Call": has_method_call,
        "Has Return Statement": has_return_statement
    }


if __name__ == "__main__":
    graph_file = sys.argv[1]
    entries = os.listdir(graph_file)
    subdirectories = [d for d in entries if os.path.isdir(os.path.join(graph_file, d))]
    for file in subdirectories:
        graph = open(file, 'r')
        graph_line = graph.readline()
        for index, value in enumerate(graph_line):
            if value == "Graph Nodes and Related Information:":
                nodes_index=index
        for j in range(nodes_index,len(graph_line)):
            if graph_line[j] == "Graph Edges:"
                break
            if graph_line[j].find("context"):
                _,context= graph_line[j].split(":")
            result = check_java_statements(context)
            state=False
            for statement, found in result.items():
                
                if found:
                    state=True
                    graph_line[j] = graph_line[j]+"operator:"+statement
            if state==False:
                graph_line[j] = graph_line[j]+"operator:"+"-1"
