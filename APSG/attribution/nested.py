import ast

def count_nesting_levels(node, level=0):
    max_level = level

    if isinstance(node, (ast.If, ast.For, ast.While)):
        max_level = max(max_level, count_nesting_levels(node.body, level + 1))
    for child in ast.iter_child_nodes(node):
        max_level = max(max_level, count_nesting_levels(child, level))

    return max_level

if __name__ == "__main__":
    code = ""
    tree = ast.parse(code)
    nesting_level = count_nesting_levels(tree)
    nested = Fasle
    if nesting_level >= 2:
        nested = True
    else:
        nested = False
