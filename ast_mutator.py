import ast
import astor
import argparse
import re
from copy import deepcopy


TAINT_MEMBERS = [
    "t_cond",
    "t_assert",
    "t_context",
    "t_assign",
    "t_aug_add",
    "t_aug_sub",
    "t_aug_mult",
]


class MutationCounter:
    def __init__(self):
        self.ctr = 0

    def get(self):
        self.ctr += 1
        return self.ctr


mutation_counter = MutationCounter()


def adbg(node):
    print(ast.dump(node, indent=4))


def apply_transformer(transformer, node):
    return ast.fix_missing_locations(transformer.visit(node))


class CtxToLoadTransformer(ast.NodeTransformer):
    def visit_Name(self, node):
        node = self.generic_visit(node)
        node.ctx = ast.Load()
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        node.ctx = ast.Load()
        return node


class CtxToStoreTransformer(ast.NodeTransformer):
    def visit_Name(self, node):
        node = self.generic_visit(node)
        node.ctx = ast.Store()
        return node

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        node.ctx = ast.Store()
        return node


class ShadowExecutionTransformer(ast.NodeTransformer):
    def __init__(self, ignore_regex):
        super().__init__()
        if ignore_regex is not None:
            self.ignore_regex = re.compile(ignore_regex)
            print(f"Got regex pattern to ignore functions: {self.ignore_regex.pattern}")
        else:
            self.ignore_regex = None
            print(f"Mutating all functions as no ignore regex was given.")

    def wrap_with(self, node, func_id):
        scope_expr = ast.Call(func=ast.Name(id=func_id, ctx=ast.Load()), args=[], keywords=[])
        new_node = ast.With([scope_expr], body=[node])
        return new_node

    def visit_FunctionDef(self, node):
        if self.ignore_regex is not None and self.ignore_regex.match(node.name):
            print(f"*NOT* Mutating: {node.name}")
            return node
        else:
            print(f"Mutating:       {node.name}")
            node = self.generic_visit(node)
            return node

    def visit_Assert(self, node):
        # skip assert statements they are handled in the AssertTransformer
        return node

    def visit_Assign(self, node):
        global mutation_counter
        node = self.generic_visit(node)

        cur_mut_ctr = ast.Constant(mutation_counter.get(), 'int')
        # Create the call to the tainted version of the right hand of the assign.
        call = ast.Call(
            func=ast.Name(id='t_assign', ctx=ast.Load()),
            args=[cur_mut_ctr, node.value],
            keywords=[]
        )
        call = apply_transformer(CtxToLoadTransformer(), call)

        node.value = call

        # This is now an Assign node
        ast.fix_missing_locations(node)
        return node

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        def change_to_tainted_call(call_id):
            global mutation_counter
            cur_mut_ctr = ast.Constant(mutation_counter.get(), 'int')
            # Create the call to the tainted version of the assign.
            call = ast.Call(
                func=ast.Name(id=call_id, ctx=ast.Load()),
                args=[cur_mut_ctr, node.target, node.value],
                keywords=[]
            )
            call = apply_transformer(CtxToLoadTransformer(), call)

            # Create the left hand side of the expression
            assign_target = deepcopy(node.target)
            if isinstance(assign_target, ast.Attribute) and isinstance(assign_target.value, ast.Name):
                assign_target.ctx = ast.Store()
            else:
                raise ValueError(f"Unknown assign_target type: {ast.dump(assign_target, indent=4)}")

            # This is now an Assign node
            newnode = ast.Assign([assign_target], call)
            ast.copy_location(newnode, node)
            ast.fix_missing_locations(newnode)
            return newnode


        if isinstance(node.op, ast.Add):
            return change_to_tainted_call('t_aug_add')
        if isinstance(node.op, ast.Sub):
            return change_to_tainted_call('t_aug_sub')
        if isinstance(node.op, ast.Mult):
            return change_to_tainted_call('t_aug_mult')
        return node


    def visit_If(self, node):
        node = self.generic_visit(node)
        assert len(node.test.ops) == 1 and len(node.test.comparators) == 1
        wrapped_test = ast.Call(
            func=ast.Name(id='t_cond', ctx=ast.Load()),
            args=[node.test],
            keywords=[]
        )
        node.test = wrapped_test
        return node


class AssertTransformer(ast.NodeTransformer):
    def visit_Assert(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Compare) and \
                len(node.test.ops) == 1 and \
                isinstance(node.test.ops[0], ast.Eq):
            call = ast.Call(func=ast.Name(id='t_assert', ctx=ast.Load()),
                            args=[node.test],
                            keywords=[])
            # Wrap the call in an Expr node, because the return value isn't used.
            newnode = ast.Expr(value=call)
            ast.copy_location(newnode, node)
            ast.fix_missing_locations(newnode)
            return newnode

        # Remember to return the original node if we don't want to change it.
        return node


def load_and_mutate(path, function_ignore_regex):
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    return tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore', help="Regex to ignore matching function names.", default=None)
    parser.add_argument('source', help="Path to source file to transform.")
    parser.add_argument('target', help="Path to output file.")

    args = parser.parse_args()

    mutated_source = load_and_mutate(args.source, args.ignore)
    with open(args.target, "wt") as f:
        f.write(astor.to_source(mutated_source))


if __name__ == "__main__":
    main()
