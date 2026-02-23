import ast

def extract_ast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "loop_depth": 0,
            "recursion_count": 0,
            "control_flow_count": 0
        }

    # 1. Control flow count
    control_flow_count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Try, ast.ExceptHandler, ast.For, ast.While)):
            control_flow_count += 1
            
    # 2. Maximum Loop depth
    def get_max_loop_depth(node, current_depth=0):
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = current_depth
            if isinstance(child, (ast.For, ast.While)):
                child_depth += 1
            max_depth = max(max_depth, get_max_loop_depth(child, child_depth))
        return max_depth

    loop_depth = get_max_loop_depth(tree)

    # 3. Recursion count
    recursion_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == func_name:
                        recursion_count += 1

    return {
        "loop_depth": loop_depth,
        "recursion_count": recursion_count,
        "control_flow_count": control_flow_count
    }
