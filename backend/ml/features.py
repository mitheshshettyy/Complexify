import ast
import io
import math
import tokenize

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


def detect_space_complexity(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Unknown"

    analyzer = _SpaceComplexityAnalyzer()
    analyzer.visit(tree)
    return analyzer.classify()


class _SpaceComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loop_depth = 0
        self.dynamic_alloc_depths = []
        self.has_halving_loop = False
        self.recursive_functions = {}

    def classify(self) -> str:
        if self.recursive_functions:
            if any(details["halving"] for details in self.recursive_functions.values()):
                return "logarithmic (recursive stack)"
            return "linear (recursive stack)"

        max_alloc_depth = max(self.dynamic_alloc_depths, default=0)
        if max_alloc_depth >= 2:
            return "quadratic"
        if max_alloc_depth == 1:
            return "linear"
        if self.has_halving_loop:
            return "logarithmic"
        return "constant"

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        if _is_halving_loop(node):
            self.has_halving_loop = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        recursion_calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == node.name:
                recursion_calls.append(child)

        if recursion_calls:
            params = [arg.arg for arg in node.args.args]
            self.recursive_functions[node.name] = {
                "halving": any(_call_halves_input(call, params) for call in recursion_calls)
            }

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._track_dynamic_allocation(node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._track_dynamic_allocation(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "extend", "insert", "add", "update"}
            and self.loop_depth > 0
        ):
            self.dynamic_alloc_depths.append(self.loop_depth)

        self.generic_visit(node)

    def _track_dynamic_allocation(self, value: ast.AST) -> None:
        alloc_depth = _allocation_depth(value, self.loop_depth)
        if alloc_depth > 0:
            self.dynamic_alloc_depths.append(alloc_depth)


def _allocation_depth(node: ast.AST, surrounding_loop_depth: int) -> int:
    if isinstance(node, ast.ListComp):
        return surrounding_loop_depth + max(1, len(node.generators))
    if isinstance(node, (ast.DictComp, ast.SetComp)):
        return surrounding_loop_depth + max(1, len(node.generators))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"list", "dict", "set", "bytearray"}:
        return max(1, surrounding_loop_depth)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if isinstance(node.left, (ast.List, ast.Tuple)) and not _is_constant_int(node.right):
            return max(1, surrounding_loop_depth)
        if isinstance(node.right, (ast.List, ast.Tuple)) and not _is_constant_int(node.left):
            return max(1, surrounding_loop_depth)
    return 0


def _is_constant_int(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int)


def _is_halving_loop(node: ast.While) -> bool:
    loop_vars = {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
    if not loop_vars:
        return False

    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id in loop_vars and _is_halving_update(target.id, child.value):
                    return True
        if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name) and child.target.id in loop_vars:
            if isinstance(child.op, (ast.FloorDiv, ast.RShift)):
                return True
    return False


def _is_halving_update(name: str, value: ast.AST) -> bool:
    if isinstance(value, ast.BinOp) and isinstance(value.left, ast.Name) and value.left.id == name:
        return isinstance(value.op, (ast.FloorDiv, ast.RShift))
    return False


def _call_halves_input(call: ast.Call, params: list[str]) -> bool:
    for arg in call.args:
        if _argument_halves_input(arg, params):
            return True
    return False


def _argument_halves_input(arg: ast.AST, params: list[str]) -> bool:
    if isinstance(arg, ast.BinOp) and isinstance(arg.left, ast.Name) and arg.left.id in params:
        return isinstance(arg.op, (ast.FloorDiv, ast.RShift))
    return False


def calculate_readability_score(code: str) -> float:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    halstead_volume = _halstead_volume(tree)
    sloc = _source_lines_of_code(code)
    comment_ratio = _comment_ratio(code)
    branch_count = _count_branch_points(tree)

    volume_term = math.log(max(halstead_volume, 1.0))
    sloc_term = math.log(max(sloc, 1))
    comment_bonus = 50.0 * math.sin(math.sqrt(2.4 * math.radians(comment_ratio)))

    maintainability = (
        171.0
        - 5.2 * volume_term
        - 0.35 * branch_count
        - 16.2 * sloc_term
        + comment_bonus
    )
    normalized = max(0.0, min(100.0, maintainability * 100.0 / 171.0))
    return round(normalized, 2)


def generate_optimization_suggestions(code: str) -> str:
    features = extract_ast_features(code)
    suggestions = []

    if features["loop_depth"] >= 2:
        suggestions.append("Reduce nested loops where possible or reuse precomputed results to lower repeated work.")
    if features["recursion_count"] > 0:
        suggestions.append("If recursion grows quickly, consider memoization or an iterative approach to reduce overhead.")
    if features["num_for_loops"] + features["num_while_loops"] > 0 and "append(" in code:
        suggestions.append("If you are building a result collection, a comprehension may simplify the loop and improve clarity.")
    if features["num_if_statements"] >= 3:
        suggestions.append("Consider extracting complex branching into smaller helper functions for easier maintenance.")

    if not suggestions:
        return "Code structure looks reasonable; optimize only after profiling identifies a real bottleneck."

    return " ".join(suggestions)


def _halstead_volume(tree: ast.AST) -> float:
    visitor = _HalsteadVisitor()
    visitor.visit(tree)
    vocabulary = len(visitor.operators) + len(visitor.operands)
    length = visitor.total_operators + visitor.total_operands
    if vocabulary == 0 or length == 0:
        return 0.0
    return float(length * math.log2(vocabulary))


class _HalsteadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.operators = set()
        self.operands = set()
        self.total_operators = 0
        self.total_operands = 0

    def visit_Name(self, node: ast.Name) -> None:
        self._add_operand(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        self._add_operand(repr(node.value))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_operand(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._add_operator("call")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._add_operator("assign")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_operator("ann_assign")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_operator(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._add_operator("return")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._add_operator(type(node.op).__name__)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._add_operator(type(node.op).__name__)
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self._add_operator(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            self._add_operator(type(op).__name__)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._add_operator("if")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._add_operator("ifexp")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add_operator("for")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._add_operator("async_for")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add_operator("while")
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._add_operator("comprehension")
        self.generic_visit(node)

    def _add_operator(self, value: str) -> None:
        self.operators.add(value)
        self.total_operators += 1

    def _add_operand(self, value: str) -> None:
        self.operands.add(value)
        self.total_operands += 1


def _count_branch_points(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Assert)):
            count += 1
        elif isinstance(node, ast.Try):
            count += len(node.handlers)
        elif isinstance(node, ast.BoolOp):
            count += max(0, len(node.values) - 1)
        elif isinstance(node, ast.comprehension):
            count += 1 + len(node.ifs)
        elif isinstance(node, ast.Match):
            count += sum(1 for case in node.cases if not _is_wildcard_case(case))
    return count


def _is_wildcard_case(case: ast.match_case) -> bool:
    return isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None and case.pattern.name is None


def _source_lines_of_code(code: str) -> int:
    return max(1, sum(1 for line in code.splitlines() if line.strip()))


def _comment_ratio(code: str) -> float:
    comment_lines = set()
    source_lines = set()

    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT:
                comment_lines.add(token.start[0])
            elif token.type not in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            }:
                source_lines.add(token.start[0])
    except tokenize.TokenError:
        return 0.0

    total_lines = len(source_lines | comment_lines)
    if total_lines == 0:
        return 0.0
    return (len(comment_lines) / total_lines) * 100.0
