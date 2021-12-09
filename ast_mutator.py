import ast
import astor
import argparse
import re
import shutil
import json
from pathlib import Path
from typing import Union
from copy import deepcopy
from mypy import api


TAINT_MEMBERS = [
    "t_combine",
    "t_wait_for_forks",
    "t_cond",
    "t_assert",
]


def adbg(node):
    print(ast.dump(node, indent=4))


def apply(transformer, node):
    return ast.fix_missing_locations(transformer.visit(node))


class MutationCounter():
    def __init__(self):
        self.ctr = 0

    def get(self):
        self.ctr += 1
        return self.ctr


class MutContainer():
    def __init__(self):
        self.mutations = []

    def add(self, mut_id, mutation):
        self.mutations.append((mut_id, mutation))

    def finalize(self):
        node = ast.parse("t_combine({})")
        dict_node = node.body[0].value.args[0]

        for mut_id, mutation_node in self.mutations:
            mut_id_node = ast.Constant(mut_id, 'int')
            dict_node.keys.append(mut_id_node)
            dict_node.values.append(mutation_node)

        return node.body[0].value



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


class ReplaceNameTransformer(ast.NodeTransformer):
    def __init__(self, name, replace_with):
        self.name = name
        self.replace_with = replace_with

    def visit_Name(self, node):
        if node.id == self.name:
            return self.replace_with
        else:
            node = self.generic_visit(node)
            return node


class Mode():
    def __init__(self, mode: str, mutations: Union[None, list[int]]):
        if mode == 'collect':
            assert mutations is None
            self.mode = 'collect'
            self.mutations = None
        elif mode == 'traditional':
            assert mutations is not None and len(mutations) == 1
            self.mode = 'traditional'
            self.mutations = set(mutations)
        elif mode == 'split-stream':
            assert mutations is not None and len(mutations) > 0
            self.mode = 'split-stream'
            self.mutations = set(mutations)
        elif mode == 'shadow':
            assert mutations is not None and len(mutations) > 0
            self.mode = 'shadow'
            self.mutations = set(mutations)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def mut_is_active(self, mut_id):
        return mut_id in self.mutations

    def is_collect(self):
        return self.mode == 'collect'

    def is_traditional(self):
        return self.mode == 'traditional'

    def is_split_stream(self):
        return self.mode == 'split-stream'

    def is_shadow(self):
        return self.mode == 'shadow'


class ShadowExecutionTransformer(ast.NodeTransformer):
    def __init__(self, mode: Mode, ignore_regex: Union[None, str]):
        super().__init__()
        self.mutation_counter = MutationCounter()
        self.mode = mode
        self.mutations = []
        if ignore_regex is not None:
            self.ignore_regex = re.compile(ignore_regex)
            # print(f"Got regex pattern to ignore functions: {self.ignore_regex.pattern}")
        else:
            self.ignore_regex = None
            # print(f"Mutating all functions as no ignore regex was given.")

    def wrap_with(self, node, func_id):
        scope_expr = ast.Call(func=ast.Name(id=func_id, ctx=ast.Load()), args=[], keywords=[])
        new_node = ast.With([scope_expr], body=[node])
        return new_node

    def visit_FunctionDef(self, node):
        if self.ignore_regex is not None and self.ignore_regex.match(node.name):
            # print(f"*NOT* Mutating: {node.name}")
            return node
        else:
            # print(f"Mutating:       {node.name}")
            node = self.generic_visit(node)
            return node

    def visit_Assert(self, node):
        # skip assert statements they are handled in the AssertTransformer
        return node

    def visit_Assign(self, node):
        global mutation_counter
        node = self.generic_visit(node)

        if self.mode.is_split_stream() or self.mode.is_shadow():
            mut_container = MutContainer()
            mut_container.add(0, node.value)

        for mutation in [
            "not right",
            "right + 1",
            "right + asdf",
            "right * 2",
            "(not right) + 1",
        ]:
            cur_mut_ctr = self.mutation_counter.get()
            mutation = ast.parse(mutation)
            if self.mode.is_traditional() and self.mode.mut_is_active(cur_mut_ctr):
                mutation = apply(ReplaceNameTransformer("right", node.value), mutation.body[0].value)
                node.value = mutation
            elif (self.mode.is_split_stream() or self.mode.is_shadow()) and self.mode.mut_is_active(cur_mut_ctr):
                mutation = apply(ReplaceNameTransformer("right", deepcopy(node.value)), mutation.body[0].value)
                mut_container.add(cur_mut_ctr, mutation)
            self.mutations.append(cur_mut_ctr)

        if self.mode.is_split_stream() or self.mode.is_shadow():
            node.value = mut_container.finalize()

        return node

    # def visit_AugAssign(self, node):
    #     node = self.generic_visit(node)
    #     def change_to_tainted_call(call_id):
    #         global mutation_counter
    #         cur_mut_ctr = ast.Constant(mutation_counter.get(), 'int')
    #         # Create the call to the tainted version of the assign.
    #         call = ast.Call(
    #             func=ast.Name(id=call_id, ctx=ast.Load()),
    #             args=[cur_mut_ctr, node.target, node.value],
    #             keywords=[]
    #         )
    #         call = apply_transformer(CtxToLoadTransformer(), call)

    #         # Create the left hand side of the expression
    #         assign_target = deepcopy(node.target)
    #         if isinstance(assign_target, ast.Attribute) and isinstance(assign_target.value, ast.Name):
    #             assign_target.ctx = ast.Store()
    #         else:
    #             raise ValueError(f"Unknown assign_target type: {ast.dump(assign_target, indent=4)}")

    #         # This is now an Assign node
    #         newnode = ast.Assign([assign_target], call)
    #         ast.copy_location(newnode, node)
    #         ast.fix_missing_locations(newnode)
    #         return newnode


    #     if isinstance(node.op, ast.Add):
    #         return change_to_tainted_call('t_aug_add')
    #     if isinstance(node.op, ast.Sub):
    #         return change_to_tainted_call('t_aug_sub')
    #     if isinstance(node.op, ast.Mult):
    #         return change_to_tainted_call('t_aug_mult')
    #     return node


    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.mode.is_shadow():
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


def collect_mutations(path, function_ignore_regex):
    # first collect all possible mutations
    mode = Mode("collect", None)
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    transformer = ShadowExecutionTransformer(mode, function_ignore_regex)
    transformer.visit(tree)
    return transformer.mutations


def generate_traditional_mutation(path, res_dir, function_ignore_regex, mut):
    # first collect all possible mutations
    mode = Mode("traditional", [mut])
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))

    res_path = res_dir/f"traditional_{mut}.py"

    with open(res_path, "wt") as f:
        f.write(astor.to_source(tree))

    mypy_filtered_dir = (res_dir/f"mypy_filtered")
    mypy_filtered_dir.mkdir(exist_ok=True)

    mypy_result = api.run([str(res_path), "--strict"])
    if mypy_result[2] != 0:
        shutil.move(res_path, mypy_filtered_dir/res_path.name)
        return False
    return True


def generate_split_stream(path, res_dir, function_ignore_regex, mutations):
    # first collect all possible mutations
    mode = Mode("split-stream", mutations)
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    tree.body.append(ast.parse(f't_wait_for_forks()').body[0])

    res_path = res_dir/f"split_stream.py"

    with open(res_path, "wt") as f:
        f.write(astor.to_source(tree))


def generate_shadow(path, res_dir, function_ignore_regex, mutations):
    # first collect all possible mutations
    mode = Mode("shadow", mutations)
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    tree.body.append(ast.parse(f't_wait_for_forks()').body[0])

    res_path = res_dir/f"shadow.py"

    with open(res_path, "wt") as f:
        f.write(astor.to_source(tree))


def load_and_mutate(path, function_ignore_regex):
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    return tree


def load_cache(cache_path):
    if cache_path is None:
        return None

    try:
        with open(cache_path, "rt") as f:
            return json.load(f)
    except Exception:
        return None


def write_cache(cache_path, data):
    if cache_path is None:
        return

    with open(cache_path, "wt") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore', help="Regex to ignore matching function names.", default=None)
    parser.add_argument('--cache', help="Mutation id cache, only use for development.", default=None)
    parser.add_argument('source', help="Path to source file to transform.")
    parser.add_argument('target', help="Path to output file.")

    args = parser.parse_args()
    result_dir = Path(args.target)
    result_dir.mkdir(parents=True)

    cache = load_cache(args.cache)

    if cache is None:
        print("Checking input file with mypy:")
        mypy_result = api.run([str(args.source)])
        print(mypy_result[0].strip())
        if mypy_result[2] != 0:
            raise ValueError("Source file does not pass type checking, fix first.")

        mutations = collect_mutations(args.source, args.ignore)
        print(mutations)


        filtered_mutations = []
        for mut in mutations:
            if generate_traditional_mutation(args.source, result_dir, args.ignore, mut):
                filtered_mutations.append(mut)
        write_cache(args.cache, filtered_mutations)
    else:
        filtered_mutations = cache
        
    print("Filtered mutations:", len(filtered_mutations), filtered_mutations)

    generate_split_stream(args.source, result_dir, args.ignore, filtered_mutations)
    generate_shadow(args.source, result_dir, args.ignore, filtered_mutations)




    # mutated_source = load_and_mutate(args.source, args.ignore)
    # with open(args.target, "wt") as f:
    #     f.write(astor.to_source(mutated_source))


if __name__ == "__main__":
    main()
