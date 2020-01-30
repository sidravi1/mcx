import ast
from typing import List, Optional, Union

import astor
import networkx as nx

from mcx.distributions import Distribution

class Argument(object):
    def __init__(self, name: str):
        self.name = name

    def to_logpdf(self):
        return ast.arg(arg=self.name, annotation=None)

    def to_sampler(self):
        return ast.arg(arg=self.name, annotation=None)


class RandVar(object):
    def __init__(
        self,
        name: str,
        distribution: Distribution,
        args: Optional[List[Union[int, float]]],
        is_returned: bool,
    ):
        self.name = name
        self.distribution = distribution
        self.args = args
        self.is_returned = is_returned

    def __str__(self):
        return "{} ~ {}".format(self.name, astor.code_gen.to_source(self.distribution))

    def to_logpdf(self):
        return ast.AugAssign(
            target=ast.Name(id="logpdf", ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution, attr="logpdf", ctx=ast.Load(),
                ),
                args=[ast.Name(id=self.name, ctx=ast.Load())],
                keywords=[],
            ),
        )

    def to_sampler(self, graph):
        add_sample_shape = True
        ancestors = nx.ancestors(graph, self.name)
        for node in ancestors:
            if isinstance(graph.nodes[node]["content"], RandVar):
                add_sample_shape = False

        args = [ast.Name(id="rng_key", ctx=ast.Load())]
        if add_sample_shape:
            args.append(ast.Name(id="sample_shape", ctx=ast.Load()))

        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution, attr="sample", ctx=ast.Load(),
                ),
                args=args,
                keywords=[],
            ),
        )


class Var(object):
    def __init__(
        self, name: str, value: Optional[Union[int, float]], is_returned: bool,
    ):
        self.name = name
        self.value = value
        self.is_returned = is_returned

    def to_logpdf(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.value
        )

    def to_sampler(self, graph):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.value
        )


class Transformation(object):
    def __init__(
        self,
        name: str,
        expression: Optional[Union[int, float]],
        args: Optional[List[Union[int, float]]],
        is_returned: bool,
    ):
        self.name = name
        self.expression = expression
        self.args = args
        self.is_returned = is_returned

    def __str__(self):
        return "{} ~ {}".format(self.name, astor.code_gen.to_source(self.expression))

    def to_logpdf(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.expression
        )

    def to_sampler(self, graph):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.expression
        )
