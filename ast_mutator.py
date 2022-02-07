import ast
import astor
import argparse
import re
import shutil
import json
import subprocess
from functools import partial
from pathlib import Path
from typing import NamedTuple, Tuple, Union, Any
from copy import deepcopy
from mypy import api
from multiprocessing import Pool


TAINT_MEMBERS = [
    "t_wrap",
    "t_combine",
    "t_gather_results",
    "t_final_exception",
    "t_cond",
    "t_assert",
]


def adbg(node):
    print(ast.dump(node, indent=4))


def apply(transformer, node):
    if type(node) == list:
        transformed = []
        for nn in node:
            res = transformer.visit(nn)
            if type(res) == list:
                res = res[0]
            transformed.append(res)
        return transformed
    else:
        return transformer.visit(node)


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

        lambda_node = ast.parse("lambda: val").body[0].value

        for mut_id, mutation_node in self.mutations:
            mut_id_node = ast.Constant(mut_id, 'int')
            mutation_val_node = apply(ReplaceNameTransformer("val", mutation_node), deepcopy(lambda_node))
            dict_node.keys.append(mut_id_node)
            dict_node.values.append(mutation_val_node)

        return node.body[0].value


class Variable():
    def __init__(self, name, getter, setter):
        self.name = name
        self.getter = getter
        self.setter = setter


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


def set_attribute(name):
    def assign(node, new):
        setattr(node, name, new)
    return assign


def assign_comparator(node, new):
    node.comparators[0] = new[0]


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

    def _wrap_with(self, node, func_id):
        scope_expr = ast.Call(func=ast.Name(id=func_id, ctx=ast.Load()), args=[], keywords=[])
        new_node = ast.With([scope_expr], body=[node])
        return new_node

    def _apply_mutations(self, node, mut_accessor, mutations, variables):
        old_node = deepcopy(node)
        if self.mode.is_split_stream() or self.mode.is_shadow():
            mut_container = MutContainer()
            mut_container.add(0, node)

        for mutation in mutations:
            if mutation is None:
                continue
            cur_mut_ctr = self.mutation_counter.get()
            mutation = ast.parse(mutation)
            mutation = mut_accessor(mutation.body[0])
            new_node = deepcopy(node)

            for var in variables:
                var.setter(mutation, apply(ReplaceNameTransformer(var.name, var.getter(new_node)), var.getter(mutation)))

            if self.mode.is_traditional() and self.mode.mut_is_active(cur_mut_ctr):
                node = mutation
            elif (self.mode.is_split_stream() or self.mode.is_shadow()) and self.mode.mut_is_active(cur_mut_ctr):
                mut_container.add(cur_mut_ctr, mutation)
            self.mutations.append(cur_mut_ctr)

        if self.mode.is_split_stream() or self.mode.is_shadow():
            node = mut_container.finalize()

        node = ast.copy_location(node, old_node)
        # node = ast.fix_missing_locations(node)

        return node

    def visit_FunctionDef(self, node):
        if self.ignore_regex is not None and self.ignore_regex.match(node.name):
            # print(f"*NOT* Mutating: {node.name}")
            return node
        else:
            # print(f"Mutating:       {node.name}")
            node = self.generic_visit(node)
            if self.mode.is_shadow():
                mutation = ast.parse("""
@t_wrap
def f():
    pass
                """).body[0]
                
                node.decorator_list.append(mutation.decorator_list[0])
            return node

    def visit_Assert(self, node):
        # skip assert statements they are handled in the AssertTransformer
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        return node

        # # do not mutate list assignments (var = [])
        # if type(node.value) == ast.List:
        #     return node

        # if type(node.value) == ast.Call and node.value.func.id == "t_combine":
        #     assign_is_mutated = True
        # else:
        #     assign_is_mutated = False

        # if self.mode.is_split_stream() or self.mode.is_shadow():
        #     mut_container = MutContainer()
        #     mut_container.add(0, node.value)

        # for mutation in [
        #     # "right != 1",
        #     "right + 1",
        #     "right * 2",
        # ]:
        #     cur_mut_ctr = self.mutation_counter.get()
        #     if assign_is_mutated:
        #         continue
        #     mutation = ast.parse(mutation)
        #     if self.mode.is_traditional() and self.mode.mut_is_active(cur_mut_ctr):
        #         mutation = apply(ReplaceNameTransformer("right", node.value), mutation.body[0].value)
        #         node.value = mutation
        #     elif (self.mode.is_split_stream() or self.mode.is_shadow()) and self.mode.mut_is_active(cur_mut_ctr):
        #         mutation = apply(ReplaceNameTransformer("right", deepcopy(node.value)), mutation.body[0].value)
        #         mut_container.add(cur_mut_ctr, mutation)
        #     self.mutations.append(cur_mut_ctr)

        # if not assign_is_mutated and (self.mode.is_split_stream() or self.mode.is_shadow()):
        #     node.value = mut_container.finalize()

        # return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)

        if len(node.ops) != 1 or len(node.comparators) != 1:
            adbg(node)
            raise NotImplementedError(f"Compare is sequence: {node}")

        # # change the compare operator:
        # #   left == right
        # # to:
        cmp_op = node.ops[0]
        mutations = [
            None if type(cmp_op) == ast.Eq    else "left == right",
            None if type(cmp_op) == ast.NotEq else "left != right",
            None if type(cmp_op) == ast.Lt    else "left <  right",
            None if type(cmp_op) == ast.LtE   else "left <= right",
            None if type(cmp_op) == ast.Gt    else "left >  right",
            None if type(cmp_op) == ast.GtE   else "left >= right",
            # None if type(cmp_op) == ast.Is    else "left is right",
        ]
        
        variables = [
            Variable("left", lambda x: x.left, set_attribute("left")),
            Variable("right", lambda x: x.comparators, set_attribute("comparators")),
        ]

        node = self._apply_mutations(node, lambda x: x.value, mutations, variables)

        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)

        # # change the operator:
        # #   left + right
        # # to:
        op = node.op
        mutations = [
            None if type(op) == ast.Add      else "left + right",
            None if type(op) == ast.Sub      else "left - right",
            None if type(op) == ast.Mult     else "left * right",
            None if type(op) == ast.Div      else "left / right",
            None if type(op) == ast.Mod      else r"left % right",
            # None if type(op) == ast.Pow      else "left ** right",
            None if type(op) == ast.LShift   else "left << right",
            None if type(op) == ast.RShift   else "left >> right",
            None if type(op) == ast.BitOr    else "left | right",
            None if type(op) == ast.BitXor   else "left ^ right",
            None if type(op) == ast.BitAnd   else "left & right",
            None if type(op) == ast.FloorDiv else "left // right",
        ]
        
        variables = [
            Variable("left", lambda x: x.left, set_attribute("left")),
            Variable("right", lambda x: x.right, set_attribute("right")),
        ]

        node = self._apply_mutations(node, lambda x: x.value, mutations, variables)

        return node


    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        return node
        

    def visit_AugAssign(self, node):
        adbg(node)
        raise NotImplementedError()
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


def generate_traditional_mutation(path, res_dir, function_ignore_regex, mut) -> Tuple[int, bool, Any]:
    # first collect all possible mutations
    mode = Mode("traditional", [mut])
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))

    try:
        source = to_source(tree)
    except Exception as exc:
        return mut, False, (1, exc, None)

    res_path = res_dir/f"traditional_{mut}.py"
    with open(res_path, "wt") as f:
        f.write(source)

    mypy_filtered_dir = (res_dir/f"mypy_filtered")
    mypy_filtered_dir.mkdir(exist_ok=True)

    mypy_result = api.run([str(res_path), "--strict"])
    if mypy_result[2] != 0:
        shutil.move(res_path, mypy_filtered_dir/res_path.name)
        print(mypy_result[0])
        return mut, False, mypy_result
    try:
        subprocess.run(['python3', res_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2)
    except subprocess.TimeoutExpired:
        shutil.move(res_path, mypy_filtered_dir/res_path.name)
        print('timed out:', mut)
        return mut, False, mypy_result
    return mut, True, mypy_result


def wrap_final_call(tree):
    final_call = tree.body[-1].value
    assert type(final_call) == ast.Call and len(final_call.args) == 0 and len(final_call.keywords) == 0
    wrapper = ast.parse("""
try:
    call()
except Exception as e:
    t_final_exception()
    raise e
                """).body[0]
    wrapper.body[0].value = final_call
    tree.body[-1] = wrapper
    return tree


def generate_split_stream(path, res_dir, function_ignore_regex, mutations):
    # first collect all possible mutations
    mode = Mode("split-stream", mutations)
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    tree = wrap_final_call(tree)

    res_path = res_dir/f"split_stream.py"

    with open(res_path, "wt") as f:
        f.write(to_source(tree))


def generate_shadow(path, res_dir, function_ignore_regex, mutations):
    # first collect all possible mutations
    mode = Mode("shadow", mutations)
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(mode, function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))

    res_path = res_dir/f"shadow_execution.py"

    with open(res_path, "wt") as f:
        f.write(to_source(tree))


def load_and_mutate(path, function_ignore_regex):
    with open(path, "rt") as f:
        tree = ast.parse(f.read())
    tree.body.insert(0, ast.parse(f'from shadow import {", ".join(TAINT_MEMBERS)}').body[0])
    tree = ast.fix_missing_locations(ShadowExecutionTransformer(function_ignore_regex).visit(tree))
    tree = ast.fix_missing_locations(AssertTransformer().visit(tree))
    return tree


def to_source(tree):
    return astor.to_source(tree, pretty_source=lambda x: ''.join(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ignore', help="Regex to ignore matching function names.", default=None)
    parser.add_argument('--active', help="Specify a specific testing type to generate, else use all.", default=None)
    parser.add_argument('source', help="Path to source file to transform.")
    parser.add_argument('target', help="Path to output file.")

    args = parser.parse_args()
    result_dir = Path(args.target)
    result_dir.mkdir(parents=True)

    # print("Checking input file with mypy:")
    mypy_result = api.run([str(args.source)])
    if mypy_result[2] != 0:
        print(mypy_result[0].strip())
        raise ValueError("Source file does not pass type checking, fix first.")

    shutil.copy(args.source, result_dir/'original.py')

    mutations = collect_mutations(args.source, args.ignore)
    print(f"There are {len(mutations)} possible mutations.")

    filtered_mutations = []
    import sys

    # generate_traditional_prepared = partial(generate_traditional_mutation, args.source, result_dir, args.ignore)
    # with Pool() as pool:
    #     for mut, keep, mypy_result in pool.imap_unordered(generate_traditional_prepared, mutations):
    for mut in mutations + [0]: # + [0] to include the unmutated version
        mut, keep, mypy_result = generate_traditional_mutation(args.source, result_dir, args.ignore, mut)
        if keep and mut != 0:
            # print(f"Keeping mutation: {mut}")
            filtered_mutations.append(mut)
        else:
            # print(f"Not keeping mutation: {mut}")
            # print(mypy_result[0])
            pass
        sys.stdout.flush()
        
    print(f"Filtered mutations:", len(filtered_mutations), sorted(filtered_mutations))

    generate_split_stream(args.source, result_dir, args.ignore, filtered_mutations)
    generate_shadow(args.source, result_dir, args.ignore, filtered_mutations)


if __name__ == "__main__":
    main()


# TODO tainting classes, see test_shadow
# TODO do not mutate __new__ functions of classes