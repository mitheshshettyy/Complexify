import ast

def extract_ast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "num_for_loops": 0,
            "num_while_loops": 0,
            "num_if_statements": 0,
            "num_functions": 0,
            "num_assignments": 0,
            "num_binary_ops": 0,
            "num_returns": 0,
            "loop_depth": 0,
            "recursion_count": 0,
            "control_flow_count": 0
        }

    features = {
        "num_for_loops": 0,
        "num_while_loops": 0,
        "num_if_statements": 0,
        "num_functions": 0,
        "num_assignments": 0,
        "num_binary_ops": 0,
        "num_returns": 0,
        "loop_depth": 0,
        "recursion_count": 0,
        "control_flow_count": 0
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            features["num_for_loops"] += 1
            features["control_flow_count"] += 1
        elif isinstance(node, ast.While):
            features["num_while_loops"] += 1
            features["control_flow_count"] += 1
        elif isinstance(node, ast.If):
            features["num_if_statements"] += 1
            features["control_flow_count"] += 1
        elif isinstance(node, ast.FunctionDef):
            features["num_functions"] += 1
        elif isinstance(node, ast.Assign) or isinstance(node, ast.AugAssign):
            features["num_assignments"] += 1
        elif isinstance(node, ast.BinOp):
            features["num_binary_ops"] += 1
        elif isinstance(node, ast.Return):
            features["num_returns"] += 1

    # Maximum Loop depth
    def get_max_loop_depth(node, current_depth=0):
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = current_depth
            if isinstance(child, (ast.For, ast.While)):
                child_depth += 1
            max_depth = max(max_depth, get_max_loop_depth(child, child_depth))
        return max_depth

    features["loop_depth"] = get_max_loop_depth(tree)

    # Recursion count
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == func_name:
                        features["recursion_count"] += 1

    return features
