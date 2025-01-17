import warnings

import jax
from jax import vmap, numpy as jnp
import numpy as np
import sympy as sp

from .base import BaseGenome
from .gene import DefaultNode, DefaultConn
from .operations import DefaultMutation, DefaultCrossover, DefaultDistance
from .utils import unflatten_conns, extract_gene_attrs, extract_gene_attrs

from ..common import (
    topological_sort,
    topological_sort_python,
    I_INF,
    attach_with_inf,
    ACT,
    AGG,
)


class DefaultGenome(BaseGenome):
    """Default genome class, with the same behavior as the NEAT-Python"""

    network_type = "feedforward"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=200,
        max_conns=100,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn(),
        mutation=DefaultMutation(),
        crossover=DefaultCrossover(),
        distance=DefaultDistance(),
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
    ):

        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene,
            conn_gene,
            mutation,
            crossover,
            distance,
            output_transform,
            input_transform,
            init_hidden_layers,
        )

    def transform(self, state, nodes, conns):
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF

        seqs = topological_sort(nodes, conn_exist)

        return seqs, nodes, conns, u_conns

    def forward(self, state, transformed, inputs):
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed
        batch_size = inputs.shape[0] if len(inputs.shape) > 1 else 1

        # Create batched initial values
        ini_vals = jnp.full((batch_size, self.max_nodes), jnp.nan)

        # Handle both batched and unbatched inputs
        if len(inputs.shape) == 1:
            inputs = inputs[None, :]  # Add batch dimension if not present

        # Set input values for all batches
        ini_vals = ini_vals.at[:, self.input_idx].set(inputs)

        nodes_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.node_gene, nodes)
        conns_attrs = vmap(extract_gene_attrs, in_axes=(None, 0))(self.conn_gene, conns)

        def cond_fun(carry):
            values, idx = carry
            return (idx < self.max_nodes) & (cal_seqs[idx] != I_INF)

        def body_func(carry):
            values, idx = carry
            i = cal_seqs[idx]

            def input_node():
                return values

            def otherwise():
                # Calculate connections
                conn_indices = u_conns[:, i]
                hit_attrs = attach_with_inf(conns_attrs, conn_indices)

                # Handle batched operations
                ins = vmap(
                    lambda vals: vmap(self.conn_gene.forward, in_axes=(None, 0, 0))(
                        state, hit_attrs, vals
                    ),
                    in_axes=0,
                )(values)

                # Calculate nodes for each batch
                z = vmap(
                    lambda ins: self.node_gene.forward(
                        state,
                        nodes_attrs[i],
                        ins,
                        is_output_node=jnp.isin(nodes[i, 0], self.output_idx),
                    )
                )(ins)

                # Update values for all batches
                new_values = values.at[:, i].set(z)
                return new_values

            values = jax.lax.cond(jnp.isin(i, self.input_idx), input_node, otherwise)
            return values, idx + 1

        final_vals, _ = jax.lax.while_loop(cond_fun, body_func, (ini_vals, 0))

        # Get outputs and handle transformation
        outputs = final_vals[:, self.output_idx]
        if self.output_transform is None:
            return outputs
        else:
            return vmap(self.output_transform)(outputs)

    def network_dict(self, state, nodes, conns):
        network = super().network_dict(state, nodes, conns)
        topo_order, topo_layers = topological_sort_python(
            set(network["nodes"]), set(network["conns"])
        )
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        return network

    def sympy_func(
        self,
        state,
        network,
        sympy_input_transform=None,
        sympy_output_transform=None,
        backend="jax",
    ):

        assert backend in ["jax", "numpy"], "backend should be 'jax' or 'numpy'"

        if sympy_input_transform is None and self.input_transform is not None:
            warnings.warn(
                "genome.input_transform is not None but sympy_input_transform is None!"
            )

        if sympy_input_transform is None:
            sympy_input_transform = lambda x: x

        if sympy_input_transform is not None:
            if not isinstance(sympy_input_transform, list):
                sympy_input_transform = [sympy_input_transform] * self.num_inputs

        if sympy_output_transform is None and self.output_transform is not None:
            warnings.warn(
                "genome.output_transform is not None but sympy_output_transform is None!"
            )

        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()
        order = network["topo_order"]

        hidden_idx = [
            i for i in network["nodes"] if i not in input_idx and i not in output_idx
        ]
        symbols = {}
        for i in network["nodes"]:
            if i in input_idx:
                symbols[-i - 1] = sp.Symbol(f"i{i - min(input_idx)}")  # origin_i
                symbols[i] = sp.Symbol(f"norm{i - min(input_idx)}")
            elif i in output_idx:
                symbols[i] = sp.Symbol(f"o{i - min(output_idx)}")
            else:  # hidden
                symbols[i] = sp.Symbol(f"h{i - min(hidden_idx)}")

        nodes_exprs = {}
        args_symbols = {}
        for i in order:

            if i in input_idx:
                nodes_exprs[symbols[-i - 1]] = symbols[
                    -i - 1
                ]  # origin equal to its symbol
                nodes_exprs[symbols[i]] = sympy_input_transform[i - min(input_idx)](
                    symbols[-i - 1]
                )  # normed i

            else:
                in_conns = [c for c in network["conns"] if c[1] == i]
                node_inputs = []
                for conn in in_conns:
                    val_represent = symbols[conn[0]]
                    # a_s -> args_symbols
                    val, a_s = self.conn_gene.sympy_func(
                        state,
                        network["conns"][conn],
                        val_represent,
                    )
                    args_symbols.update(a_s)
                    node_inputs.append(val)
                nodes_exprs[symbols[i]], a_s = self.node_gene.sympy_func(
                    state,
                    network["nodes"][i],
                    node_inputs,
                    is_output_node=(i in output_idx),
                )
                args_symbols.update(a_s)

                if i in output_idx and sympy_output_transform is not None:
                    nodes_exprs[symbols[i]] = sympy_output_transform(
                        nodes_exprs[symbols[i]]
                    )

        input_symbols = [symbols[-i - 1] for i in input_idx]
        reduced_exprs = nodes_exprs.copy()
        for i in order:
            reduced_exprs[symbols[i]] = reduced_exprs[symbols[i]].subs(reduced_exprs)

        output_exprs = [reduced_exprs[symbols[i]] for i in output_idx]

        lambdify_output_funcs = [
            sp.lambdify(
                input_symbols + list(args_symbols.keys()),
                exprs,
                modules=[backend, AGG.sympy_module(backend), ACT.sympy_module(backend)],
            )
            for exprs in output_exprs
        ]

        fixed_args_output_funcs = []
        for i in range(len(output_idx)):

            def f(inputs, i=i):
                return lambdify_output_funcs[i](*inputs, *args_symbols.values())

            fixed_args_output_funcs.append(f)

        forward_func = lambda inputs: jnp.array(
            [f(inputs) for f in fixed_args_output_funcs]
        )

        return (
            symbols,
            args_symbols,
            input_symbols,
            nodes_exprs,
            output_exprs,
            forward_func,
        )

    def visualize(
        self,
        network,
        rotate=0,
        reverse_node_order=False,
        color=("yellow", "white", "blue"),
        edgecolors="k",
        arrowstyle="->",
        arrowsize=3,
        edge_color=(0.3, 0.3, 0.3),
        save_path="network.svg",
        save_dpi=50,  # Reduced default DPI
        figure_size=(10, 7),  # Control figure size explicitly
        **kwargs,
    ):
      
        import networkx as nx
        from matplotlib import pyplot as plt
        import numpy as np

        conns_list = list(network["conns"])
        input_idx = self.get_input_idx()
        output_idx = self.get_output_idx()

        topo_order, topo_layers = network["topo_order"], network["topo_layers"]
        node2layer = {
            node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
        }
        if reverse_node_order:
            topo_order = topo_order[::-1]

        G = nx.DiGraph()

        # Create node labels dictionary
        node_labels = {}

        # Add nodes with activation function info
        for node in topo_order:
            node_data = network["nodes"][node]
            activation = node_data.get("act", "none")  # Get activation function name
            
            # Create concise label
            node_labels[node] = f"{node}\n{activation}"
            
            # Add node with appropriate color
            if node in input_idx:
                G.add_node(node, subset=node2layer[node], color=color[0])
            elif node in output_idx:
                G.add_node(node, subset=node2layer[node], color=color[2])
            else:
                G.add_node(node, subset=node2layer[node], color=color[1])

        # Add edges
        for conn in conns_list:
            G.add_edge(conn[0], conn[1])

        pos = nx.multipartite_layout(G)

        # Rotation function
        def rotate_layout(pos, angle):
            angle_rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
            return {
                node: (
                    cos_angle * x - sin_angle * y,
                    sin_angle * x + cos_angle * y,
                )
                for node, (x, y) in pos.items()
            }

        rotated_pos = rotate_layout(pos, rotate)
        node_colors = [n["color"] for n in G.nodes.values()]

        # Clear any existing plots and set figure size
        plt.clf()
        plt.figure(figsize=figure_size)

        # Remove with_labels from kwargs if present
        kwargs.pop('with_labels', None)

        # Draw the network
        nx.draw(
            G,
            pos=rotated_pos,
            node_color=node_colors,
            node_size=1500,  # Reduced node size
            edgecolors=edgecolors,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            edge_color=edge_color,
            with_labels=True,
            labels=node_labels,
            font_size=7,  # Smaller font size
            font_weight="bold",
            **kwargs,
        )

        # Save with optimized settings
        plt.savefig(
            save_path, 
            dpi=save_dpi, 
            bbox_inches="tight",
            format='svg' if save_path.endswith('.svg') else None
        )
        plt.close()

    # def visualize(
    #     self,
    #     network,
    #     rotate=0,
    #     reverse_node_order=False,
    #     size=(300, 300, 300),
    #     color=("yellow", "white", "blue"),
    #     with_labels=False,
    #     edgecolors="k",
    #     arrowstyle="->",
    #     arrowsize=3,
    #     edge_color=(0.3, 0.3, 0.3),
    #     save_path="network.svg",
    #     save_dpi=800,
    #     **kwargs,
    # ):
    #     import networkx as nx
    #     from matplotlib import pyplot as plt

    #     print("GENE GATE 1")

    #     conns_list = list(network["conns"])
    #     input_idx = self.get_input_idx()
    #     output_idx = self.get_output_idx()

    #     print("GENE GATE 2")

    #     topo_order, topo_layers = network["topo_order"], network["topo_layers"]
    #     node2layer = {
    #         node: layer for layer, nodes in enumerate(topo_layers) for node in nodes
    #     }
    #     if reverse_node_order:
    #         topo_order = topo_order[::-1]

    #     G = nx.DiGraph()

    #     print("GENE GATE 3")

    #     if not isinstance(size, tuple):
    #         size = (size, size, size)
    #     if not isinstance(color, tuple):
    #         color = (color, color, color)

    #     print("GENE GATE 3.1")

    #     print("topo order:", topo_order)
    #     print("topo layers:", topo_layers)
    #     print("node2layer", node2layer)
    #     for node in topo_order:
    #         print("node:", node, "inputidx:", input_idx, "outputidx:", output_idx)
    #         if node in input_idx:
    #             G.add_node(node, subset=node2layer[node], size=size[0], color=color[0])
    #         elif node in output_idx:
    #             print("subset check:", node2layer[node])
    #             G.add_node(node, subset=node2layer[node], size=size[2], color=color[2])
    #         else:
    #             G.add_node(node, subset=node2layer[node], size=size[1], color=color[1])
    #         print("node out success: ", node)

    #     print("GENE GATE 3.2")
    #     for conn in conns_list:
    #         G.add_edge(conn[0], conn[1])
    #     pos = nx.multipartite_layout(G)

    #     def rotate_layout(pos, angle):
    #         angle_rad = np.deg2rad(angle)
    #         cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    #         rotated_pos = {}
    #         for node, (x, y) in pos.items():
    #             rotated_pos[node] = (
    #                 cos_angle * x - sin_angle * y,
    #                 sin_angle * x + cos_angle * y,
    #             )
    #         return rotated_pos

    #     print("GENE GATE 3.3")
    #     rotated_pos = rotate_layout(pos, rotate)

    #     print("GENE GATE 3.4")
    #     node_sizes = [n["size"] for n in G.nodes.values()]
    #     node_colors = [n["color"] for n in G.nodes.values()]

    #     print("GENE GATE 4")
    #     nx.draw(
    #         G,
    #         pos=rotated_pos,
    #         node_size=node_sizes,
    #         node_color=node_colors,
    #         with_labels=with_labels,
    #         edgecolors=edgecolors,
    #         arrowstyle=arrowstyle,
    #         arrowsize=arrowsize,
    #         edge_color=edge_color,
    #         **kwargs,
    #     )
    #     plt.savefig(save_path, dpi=save_dpi)
    #     plt.close()
