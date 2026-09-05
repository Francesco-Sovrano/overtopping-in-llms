from typing import List, Dict, Union, Tuple, Literal, Optional, Set

from einops import einsum
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import math
from collections import defaultdict


class Node:
	"""
	A node in our computational graph. The in_hook is the TL hook into its inputs, 
	while the out_hook gets its outputs.
	"""
	name: str
	layer: int
	in_hook: str
	out_hook: str
	index: Tuple
	parents: Set['Node']
	parent_edges: Set['Edge']
	children: Set['Node']
	child_edges: Set['Edge']
	in_graph: bool
	score: Optional[float]
	neurons: Optional[torch.Tensor]
	neurons_scores: Optional[torch.Tensor]
	qkv_inputs: Optional[List[str]]

	def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, 
				 graph:'Graph', qkv_inputs: Optional[List[str]]=None):
		self.in_hook = in_hook
		self.name = name
		self.layer = layer
		self.in_hook = in_hook
		self.out_hook = out_hook 
		self.index = index
		self.graph = graph
		self.parents = set()
		self.children = set()
		self.parent_edges = set()
		self.child_edges = set()
		self.qkv_inputs = qkv_inputs
	
	def __repr__(self):
		return f'Node({self.name}, in_graph: {self.in_graph})'
	
	def __hash__(self):
		return hash(self.name)
	
	# Nodes just report back their in_graph/score/neurons_in_graph/neurons_scores status from the graph
	@property
	def in_graph(self):
		return self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)]

	@in_graph.setter
	def in_graph(self, value):
		self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)] = value

	@property
	def score(self):
		if self.graph.nodes_scores is None:
			return None
		return self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)]

	@score.setter
	def score(self, value):
		if self.graph.nodes_scores is None:
			raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
		self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)] = value
		
	@property
	def neurons(self):
		if self.graph.neurons is None:
			return None
		return self.graph.neurons[self.graph.forward_index(self, attn_slice=False)]

	@neurons.setter
	def neurons(self, value):
		if self.graph.neurons is None:
			raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
		self.graph.neurons[self.graph.forward_index(self, attn_slice=False)] = value
		
	@property
	def neurons_scores(self):
		if self.graph.neurons_scores is None:
			return None
		return self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)]

	@neurons_scores.setter
	def neurons_scores(self, value):
		if self.graph.neurons_scores is None:
			raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
		self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)] = value

class LogitNode(Node):
	def __init__(self, n_layers:int, graph:'Graph'):
		name = 'logits'
		index = slice(None)
		# was: f"blocks.{n_layers - 1}.hook_resid_post"
		super().__init__(name, n_layers - 1, "ln_final.hook_normalized", "", index, graph)
		
	@property
	def in_graph(self):
		return True

	@in_graph.setter
	def in_graph(self, value):
		raise ValueError(f"Cannot set in_graph for logits node (always True)")
		
class MLPNode(Node):
	def __init__(self, layer: int, graph: 'Graph'):
		name = f'm{layer}'
		index = slice(None)
		super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index, graph)

class AttentionNode(Node):
	head: int
	def __init__(self, layer:int, head:int, graph: 'Graph'):
		name = f'a{layer}.h{head}' 
		self.head = head
		index = (slice(None), slice(None), head) 
		super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, graph, qkv_inputs=[f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
	def __init__(self, graph: 'Graph'):
		name = 'input' 
		index = slice(None) 
		super().__init__(name, 0, '', "hook_embed", index, graph)

class Edge:
	"""An Edge in a graph. 
	Attributes:
		name: (str): the edge's name, given as [PARENT]->[CHILD]<[OPTIONAL QKV>]; the latter applies only if [CHILD] is an AttentionNode
		parent: (Node): the parent node of the edge
		child: (Node): the child node of the edge
		hook: (str): the hook into the child node
		index: (Tuple): the index of the child node (only really relevant for AttentionNodes)
		score: (Optional[float]): the score of the edge (given by an attribution method)
		in_graph: (bool): whether the edge is in the graph or not"""

	name: str
	parent: Node 
	child: Node 
	hook: str
	index: Tuple
	graph: 'Graph'
	def __init__(self, graph: 'Graph', parent: Node, child: Node, qkv:Optional[Literal["q", "k", "v"]]=None):
		self.graph = graph
		self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
		self.parent = parent 
		self.child = child
		self.qkv = qkv
		self.matrix_index = (graph.forward_index(parent, attn_slice=False), graph.backward_index(child, qkv, attn_slice=False))
				
		if isinstance(child, AttentionNode):
			if qkv is None:
				raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
			self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
			self.index = (slice(None), slice(None), child.head)
		else:
			self.index = child.index
			self.hook = child.in_hook

	def __repr__(self):
		return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'
	
	def __hash__(self):
		return hash(self.name)
	
	@property
	def score(self):
		return self.graph.scores[self.matrix_index]
	
	@score.setter
	def score(self, value):
		self.graph.scores[self.matrix_index] = value
	
	@property
	def in_graph(self):
		return self.graph.in_graph[self.matrix_index]
	
	@in_graph.setter
	def in_graph(self, value):
		self.graph.in_graph[self.matrix_index] = value
		
class GraphConfig(dict):
	def __init__(self, *args, **kwargs):
		super(GraphConfig, self).__init__(*args, **kwargs)
		self.__dict__ = self

class Graph:
	"""
	Represents a graph that consists of nodes and edges.

	Attributes:
		nodes (Dict[str, Node]): A dictionary of nodes in the graph, where the key is the node name and the value is the node object.
		edges (Dict[str, Edge]): A dictionary of edges in the graph, where the key is the edge name and the value is the edge object.
		n_forward (int): The number of forward nodes in the graph, i.e. the # of nodes whose output activations we care about
		n_backward (int): The number of backward nodes/indices in the graph, i.e. the # of nodes whose input gradients we care about. Note that attention heads have 3 inputs that need to be dealt with during a backward pass
		cfg (HookedTransformerConfig): The configuration object for the graph.
	"""
	nodes: Dict[str, Node]  # Maps from node names ('input', 'a0.h0', 'm0', 'logits', etc.) to Node objects
	edges: Dict[str, Edge]  # Maps from edge names ('input->a0.h0', 'a0.h0->m0', etc.) to Edge objects. Attn edges are denoted as 'input->a0.h0<q>', 'input->a0.h0<k>', 'input->a0.h0<v>'
	n_forward: int  # the number of forward (source) nodes
	n_backward: int  # the number of backward (destination) nodes
	scores: torch.Tensor  # (n_forward, n_backward) tensor of edge scores
	in_graph: torch.Tensor  # (n_forward, n_backward) tensor of whether the edge is in the graph
	neurons_scores: Optional[torch.Tensor]  # (n_forward, d_model) tensor of neuron scores for each forward node. If a neuron's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
	neurons_in_graph: Optional[torch.Tensor]  # (n_forward, d_model) tensor of whether the neuron is in the graph
	nodes_scores: Optional[torch.Tensor]  # (n_forward) tensor of source node scores. If None, nodes have no scores. If a node's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
	nodes_in_graph: torch.Tensor  # (n_forward) tensor of whether the (source) node is in the graph
	forward_to_backward: torch.Tensor
	real_edge_mask: torch.Tensor   # (n_forward, n_backward) tensor of whether the edge is real (some edges are not real, e.g. m10->m2)
	cfg: GraphConfig

	def __init__(self):
		self.nodes = {}
		self.edges = {}
		self.n_forward = 0
		self.n_backward = 0
		

	def copy(self):
		"""
		Deep copy of the Graph.
		- Rebuilds nodes/edges from cfg (so all Node/Edge objects point to the new graph)
		- Clones all tensor fields (no shared storage)
		"""
		# Recreate the structure with the same configuration and feature flags
		new = Graph.from_model(
			dict(self.cfg),
			neuron_level=(self.neurons_in_graph is not None),
			node_scores=(self.nodes_scores is not None),
		)

		# Clone core tensors (preserve dtypes/devices; avoid shared storage)
		new.forward_to_backward = self.forward_to_backward.clone()
		new.real_edge_mask      = self.real_edge_mask.clone()
		new.scores              = self.scores.clone()
		new.in_graph            = self.in_graph.clone()
		new.nodes_in_graph      = self.nodes_in_graph.clone()

		# Optional tensors
		if self.nodes_scores is not None:
			new.nodes_scores = self.nodes_scores.clone()
		else:
			new.nodes_scores = None

		if self.neurons_in_graph is not None:
			new.neurons_in_graph = self.neurons_in_graph.clone()
		else:
			new.neurons_in_graph = None

		if self.neurons_scores is not None:
			new.neurons_scores = self.neurons_scores.clone()
		else:
			new.neurons_scores = None

		return new

	def _ensure_edge_index_map(self):
		# Build a reverse map (i,j) -> edge_name once
		if not hasattr(self, "_edge_index_to_name"):
			idx2name = {}
			for name, e in self.edges.items():
				i, j = e.matrix_index
				idx2name[(int(i), int(j))] = name
			self._edge_index_to_name = idx2name

	def _forward_index_to_name(self):
		# Build a forward-index -> node name array once
		if not hasattr(self, "_fwd_idx_to_name"):
			idx2name = [None] * self.n_forward
			for node in self.nodes.values():
				try:
					i = self.forward_index(node, attn_slice=False)
					idx2name[int(i)] = node.name
				except Exception:
					pass
			self._fwd_idx_to_name = idx2name

	def add_edge(self, parent:Node, child:Node, qkv:Optional[Literal["q", "k", "v"]]=None):
		edge = Edge(self, parent, child, qkv)
		self.real_edge_mask[edge.matrix_index] = True
		self.edges[edge.name] = edge
		parent.children.add(child)
		parent.child_edges.add(edge)
		child.parents.add(parent)
		child.parent_edges.add(edge)
		
	def prev_index(self, node: Node) -> Union[int, slice]:
		"""Return the forward index before which all nodes contribute to the input of the given node
		Args:
			node (Node): The node to get the prev forward index of
		Returns:
			Union[int, slice]: an index representing the prev forward index of the node
		"""
		if isinstance(node, InputNode):
			return 0
		elif isinstance(node, LogitNode):
			return self.n_forward
		elif isinstance(node, MLPNode):
			if self.cfg['parallel_attn_mlp']:
				return 1 + node.layer * (self.cfg['n_heads'] + 1)
			else:
				return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
		elif isinstance(node, AttentionNode):
			i =  1 + node.layer * (self.cfg['n_heads'] + 1)
			return i
		else:
			raise ValueError(f"Invalid node: {node} of type {type(node)}")
		
	@classmethod
	def _forward_index(cls, cfg, node_name:str, attn_slice:bool=False) -> int:
		"""Given a model's config and a node specification, return the forward (source) index of the node in the graph. The forward index is the index of the node in the forward pass of the model, which is used to index into the graph's tensors.
		
		Args:
			cfg (_type_): a (HookedTransformer) config object
			node_name (str): Name of the node: 'input', 'logits', 'm0', 'a0.h0', etc.
			attn_slice (bool, optional): _description_. Defaults to False.

		Returns:
			int: the forward index
		"""
		if node_name == 'input':
			return 0
		elif node_name == 'logits':
			raise ValueError("Logits has no forward index (it's a sink).")
		elif node_name[0] == 'm':
			layer = int(node_name[1:])
			return 1 + layer * (cfg.n_heads + 1) + cfg.n_heads
		elif node_name[0] == 'a':
			layer, head = node_name.split('.')
			layer = int(layer[1:])
			head = int(head[1:])
			i =  1 + layer * (cfg.n_heads + 1)
			return slice(i, i + cfg.n_heads) if attn_slice else i + head
		else:
			raise ValueError(f"Invalid node: {node_name}") 

	def forward_index(self, node:Node, attn_slice=True) -> int:
		return Graph._forward_index(self.cfg, node.name, attn_slice)        

	@classmethod
	def _backward_index(cls, cfg, node_name:str, qkv=None, attn_slice=False) -> int:
		"""Given a model's config and a node specification, return the backward (destination) index of the node in the graph. The backward index is the index of the node in the backward pass of the model, which is used to index into the graph's tensors.

		Args:
			cfg (_type_): A (HookedTransformer) config object
			node_name (str): Name of the node: 'input', 'logits', 'm0', 'a0.h0', etc.
			qkv (_type_, optional): Whether the destination (for attention heads) is the q/k/v input. Defaults to None.
			attn_slice (bool, optional): _description_. Defaults to False.

		Returns:
			int: the backward index
		"""
		if node_name == 'input':
			raise ValueError(f"No backward for input node")
		elif node_name == 'logits':
			return -1
		elif node_name[0] == 'm':
			layer = int(node_name[1:])
			return (layer) * (3 * cfg['n_heads'] + 1) + 3 * cfg['n_heads']
		elif node_name[0] == 'a':
			assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
			layer, head = node_name.split('.')
			layer = int(layer[1:])
			head = int(head[1:])
			i = layer * (3 * cfg['n_heads'] + 1) + ('qkv'.index(qkv) * cfg['n_heads'])
			return slice(i, i + cfg['n_heads']) if attn_slice else i + head
		else:
			raise ValueError(f"Invalid node: {node_name}")
		
	def backward_index(self, node:Node, qkv=None, attn_slice=True) -> int:
		return Graph._backward_index(self.cfg, node.name, qkv, attn_slice)
		
	def count_included_neurons(self) -> int:
		if self.neurons_in_graph is None:
			return None
		return self.neurons_in_graph.sum().item()
	
	def reset(self, empty=True):
		"""Resets the graph, setting everything to zero. If empty is False, sets everything to True instead.
		Args:
			empty (bool, optional): If true, removes everything from graph; otherwise adds everything. Defaults to True.
		"""
		if empty:
			self.nodes_in_graph *= False
			self.in_graph *= False
			if self.neurons_in_graph is not None:
				self.neurons_in_graph *= False
		else:
			self.nodes_in_graph[:] = True
			self.in_graph[:] = True
			self.in_graph &= self.real_edge_mask
			if self.neurons_in_graph is not None:
				self.neurons_in_graph[:] = True
				
	def zero_out_attention_neuron_scores(graph):
		if graph.neurons_scores is None:
			return
		for node in graph.nodes.values():
			if isinstance(node, AttentionNode):
				fi = graph.forward_index(node, attn_slice=False)
				graph.neurons_scores[fi, :] = 0

	def zero_out_attention_node_scores(graph):
		if graph.nodes_scores is None:
			return
		for node in graph.nodes.values():
			if isinstance(node, AttentionNode):
				fi = graph.forward_index(node, attn_slice=False)
				graph.nodes_scores[fi] = 0

	def zero_out_attention_to_attention_edge_scores(graph):
		for edge in graph.edges.values():
			if isinstance(edge.parent, AttentionNode) and isinstance(edge.child, AttentionNode):
				graph.scores[edge.matrix_index] = 0

	def saturate_selected_nodes(
		self,
		include_edges = "both",   # "in" | "out" | "both"
		include_neurons = True
	):
		"""
		If nodes_in_graph[i] == True, then:
		  - include all its neurons (row i) in neurons_in_graph (if present),
		  - include all outgoing edges from that node,
		  - include all incoming edges to that node,
		all restricted to real edges.
		"""
		sel = self.nodes_in_graph.clone()                              # [n_forward] (bool)

		# (1) All neurons in selected nodes
		if include_neurons and (self.neurons_in_graph is not None):
			self.neurons_in_graph[sel, :] = True                       # set entire rows True

		# (2) Outgoing edges from selected nodes
		if include_edges in ("out", "both"):
			# broadcast node mask on forward dim and intersect with real edges
			self.in_graph |= (sel.view(-1, 1) & self.real_edge_mask)   # [n_forward, n_backward]

		# (3) Incoming edges to selected nodes
		if include_edges in ("in", "both"):
			# Which backward slots (destinations) correspond to the selected forward nodes?
			# forward_to_backward: [n_forward, n_backward] (bool)
			# dst_mask[b] = True if ANY selected forward node maps to backward b
			dst_mask = (self.forward_to_backward & sel.view(-1, 1)).any(dim=0)  # [n_backward] (bool)
			self.in_graph |= (self.real_edge_mask & dst_mask.view(1, -1)) 

	def _rank_indices_1d(self, scores, valid_mask, n, *, absolute = True, include_zero_scores = False):
		"""
		Shared 1D ranking helper.

		scores:     tensor of scores (any shape, will be flattened)
		valid_mask: bool tensor, same shape as scores; True = part of population
		n:          requested top-n
		absolute:   rank by |score| if True, else by score
		include_zero_scores: whether exact zero scores are eligible for top-N selection

		Returns a dict with:
			- idxs          : 1D LongTensor of selected flat indices (len <= n)
			- total_entities: scores.numel()
			- num_considered: valid_mask.sum()
			- num_scored    : count of finite scores among considered
			- score_range   : (min, max) among finite considered scores (post-|·| if absolute)
			- threshold     : last selected score (original signed value), or None
		"""
		scores_flat = scores.view(-1)
		valid_flat  = valid_mask.view(-1).to(dtype=torch.bool)

		assert scores_flat.shape == valid_flat.shape

		total_entities = int(scores_flat.numel())
		num_considered = int(valid_flat.sum().item())

		if num_considered == 0:
			return {
				"idxs": torch.empty(0, dtype=torch.long, device=scores.device),
				"total_entities": total_entities,
				"num_considered": 0,
				"num_scored": 0,
				"score_range": None,
				"threshold": None,
			}

		considered = scores_flat[valid_flat]
		finite_mask = torch.isfinite(considered)
		eligible_mask = finite_mask if include_zero_scores else (finite_mask & (considered != 0))
		num_scored = int(eligible_mask.sum().item())

		if num_scored == 0:
			return {
				"idxs": torch.empty(0, dtype=torch.long, device=scores.device),
				"total_entities": total_entities,
				"num_considered": num_considered,
				"num_scored": 0,
				"score_range": None,
				"threshold": None,
			}

		eligible_scores = considered[eligible_mask]
		range_vals = eligible_scores.abs() if absolute else eligible_scores
		score_range = (float(range_vals.min().item()), float(range_vals.max().item()))

		# ranking tensor – only eligible values are considered.
		# By default, exact zeros are excluded to preserve historical behavior.
		rank = considered.clone()
		rank[~eligible_mask] = 0.0
		if absolute:
			rank = rank.abs()
		rank[~eligible_mask] = float('-inf')

		k = min(int(n), num_scored)
		_, local_idxs = torch.topk(rank, k=k, largest=True, sorted=True)

		# map back to flat indices in scores
		valid_idx = torch.nonzero(valid_flat, as_tuple=False).view(-1)
		idxs = valid_idx[local_idxs]

		threshold = float(scores_flat[idxs[-1]].item())

		return {
			"idxs": idxs,
			"total_entities": total_entities,
			"num_considered": num_considered,
			"num_scored": num_scored,
			"score_range": score_range,
			"threshold": threshold,
		}

	def get_topn(self, n, level: Literal['edge', 'node', 'neuron'] = 'node', absolute = True, include_special = False, return_scores = False, return_metadata = False, include_zero_scores = False):
		"""
		Return the top-n components by score WITHOUT mutating the graph.

		level='edge'   -> list of edge names (or (name, score))
		level='node'   -> list of node names (or (name, score))
		level='neuron' -> list of (node_name, neuron_idx) (or ((node_name, neuron_idx), score))

		absolute:        rank by |score|.
		include_special: include 'input' and 'logits' in node/neuron results.
		return_scores:   when True, return (identifier, float_score) tuples.
		return_metadata: when True, also return a metadata dict as a second value.
		include_zero_scores: when True, finite exact-zero scores can be selected.
		"""
		if level not in {'edge', 'node', 'neuron'}:
			raise ValueError(f"Invalid level: {level}")

		meta = {
			"level": level,
			"absolute": bool(absolute),
			"include_special": bool(include_special),
			"requested_n": int(n),
			"selected_n": 0,
			"num_scored": 0,
			"num_considered": 0,
			"total_entities": 0,
			"was_clipped": False,
			"score_range": None,
			"threshold": None,
			"empty_reason": None,
			"indices": [],
			"include_zero_scores": bool(include_zero_scores),
		}
		meta["d_model"] = int(self.cfg.get("d_model")) if self.cfg.get("d_model") is not None else None

		def _finish(items):
			meta["selected_n"] = len(items)
			meta["was_clipped"] = meta["selected_n"] < meta["requested_n"]
			if return_metadata:
				return items, meta
			return items

		# --------------------- EDGE LEVEL ---------------------
		if level == 'edge':
			self._ensure_edge_index_map()

			scores = self.scores
			valid_mask = self.real_edge_mask

			info = self._rank_indices_1d(scores, valid_mask, n, absolute=absolute, include_zero_scores=include_zero_scores)

			meta["total_entities"] = info["total_entities"]
			meta["num_considered"] = info["num_considered"]
			meta["num_scored"] = info["num_scored"]
			meta["score_range"] = info["score_range"]
			meta["threshold"] = info["threshold"]

			if meta["num_considered"] == 0:
				meta["empty_reason"] = "no real edges"
				return _finish([])

			if meta["num_scored"] == 0:
				meta["empty_reason"] = "no scored edges yet"
				return _finish([])

			flat_scores = scores.view(-1)
			res = []
			label_score = {}

			for idx in info["idxs"].tolist():
				val = float(flat_scores[idx])
				if not math.isfinite(val):
					continue
				i = idx // self.n_backward
				j = idx % self.n_backward
				name = self._edge_index_to_name.get((int(i), int(j)))
				if name is None:
					continue
				res.append((name, val) if return_scores else name)
				label_score[name] = val
				meta["indices"].append(int(idx))

			meta["edge_label_score"] = label_score
			return _finish(res)

		# --------------------- NODE LEVEL ---------------------
		if level == 'node':
			self._forward_index_to_name()
			scores = self.nodes_scores

			meta["total_entities"] = int(scores.numel())

			# valid population, respecting special nodes
			valid_mask = torch.zeros_like(scores, dtype=torch.bool)
			for i, name in enumerate(self._fwd_idx_to_name):
				if name is None:
					continue
				if not include_special and name in ("input", "logits"):
					continue
				valid_mask[i] = True

			info = self._rank_indices_1d(scores, valid_mask, n, absolute=absolute, include_zero_scores=include_zero_scores)

			meta["num_considered"] = info["num_considered"]
			meta["num_scored"] = info["num_scored"]
			meta["score_range"] = info["score_range"]
			meta["threshold"] = info["threshold"]

			if meta["num_considered"] == 0:
				meta["empty_reason"] = "no nodes to consider (special nodes filtered?)"
				return _finish([])

			if meta["num_scored"] == 0:
				meta["empty_reason"] = "no scored nodes yet"
				return _finish([])

			flat_scores = scores.view(-1)
			out = []
			label_score = {}

			for idx in info["idxs"].tolist():
				v = float(flat_scores[idx])
				if not math.isfinite(v):
					continue
				name = self._fwd_idx_to_name[int(idx)]
				out.append((name, v) if return_scores else name)
				label_score[name] = v
				meta["indices"].append(int(idx))

			meta["node_label_score"] = label_score
			return _finish(out)

		# --------------------- NEURON LEVEL ---------------------
		# level == 'neuron'
		if self.neurons_scores is None:
			raise RuntimeError("neurons_scores is None; cannot compute top-N neurons.")

		self._forward_index_to_name()
		ns = self.neurons_scores
		d_model = self.cfg['d_model']

		meta["total_entities"] = int(ns.numel())

		# population mask (optionally drop special nodes)
		valid_mask = torch.ones_like(ns, dtype=torch.bool)
		if not include_special:
			for i, name in enumerate(self._fwd_idx_to_name):
				if name in ("input", "logits"):
					valid_mask[i, :] = False

		info = self._rank_indices_1d(ns, valid_mask, n, absolute=absolute, include_zero_scores=include_zero_scores)

		meta["num_considered"] = info["num_considered"]
		meta["num_scored"] = info["num_scored"]
		meta["score_range"] = info["score_range"]
		meta["threshold"] = info["threshold"]

		if meta["num_considered"] == 0:
			meta["empty_reason"] = "no neurons to consider (special nodes filtered?)"
			return _finish([])

		if meta["num_scored"] == 0:
			meta["empty_reason"] = "no scored neurons yet"
			return _finish([])

		flat_scores = ns.view(-1)
		out = []
		label_score = {}
		mlp_neurons_dict = defaultdict(list)

		for idx in info["idxs"].tolist():
			v = float(flat_scores[idx])
			if not math.isfinite(v):
				continue
			i = idx // d_model
			h = idx % d_model
			node_name = self._fwd_idx_to_name[int(i)]
			ident = str((node_name, int(h)))

			out.append((ident, v) if return_scores else ident)
			label_score[ident] = v
			mlp_neurons_dict[node_name].append(int(h))
			meta["indices"].append(int(idx))

		meta["neuron_label_score"] = label_score
		meta["neurons"] = mlp_neurons_dict
		return _finish(out)
	
	def apply_topn(self, n, absolute = True, level: Literal['edge', 'node', 'neuron'] = 'node', reset = True, prune = True, include_zero_scores = False):
		"""
		Sets the graph to contain only the top-n components.
		"""
		if reset:
			self.reset()

		if level == 'neuron':
			scored_neurons = ~torch.isnan(self.neurons_scores)

			info = self._rank_indices_1d(
				self.neurons_scores,
				valid_mask=scored_neurons,
				n=n,
				absolute=absolute,
				include_zero_scores=include_zero_scores,
			)
			idxs = info["idxs"]

			flat_neurons = self.neurons_in_graph.view(-1)
			flat_neurons[:] = False
			flat_neurons[idxs] = True
			# unscored neurons must also be re-added
			flat_neurons[~scored_neurons.view(-1)] = True

			if reset:
				self.nodes_in_graph |= self.neurons_in_graph.any(dim=1)
				self.in_graph |= (self.nodes_in_graph.view(-1, 1) & self.real_edge_mask)

			print(
				'[dbg] neurons_in_graph:',
				self.neurons_in_graph.sum().item(),
				'nodes_in_graph:',
				self.nodes_in_graph.sum().item(),
				'edges_in_graph:',
				self.in_graph.sum().item(),
			)

		elif level == 'node':
			scored_nodes = ~torch.isnan(self.nodes_scores)

			info = self._rank_indices_1d(
				self.nodes_scores,
				valid_mask=scored_nodes,
				n=n,
				absolute=absolute,
				include_zero_scores=include_zero_scores,
			)
			idxs = info["idxs"]

			flat_nodes = self.nodes_in_graph.view(-1)
			flat_nodes[:] = False
			flat_nodes[idxs] = True
			# unscored nodes must also be re-added
			flat_nodes[~scored_nodes.view(-1)] = True

			self.saturate_selected_nodes(include_edges="both", include_neurons=True)

			if reset:
				self.in_graph |= (self.nodes_in_graph.view(-1, 1) & self.real_edge_mask)

			print(
				'[dbg] nodes_in_graph:',
				self.nodes_in_graph.sum().item(),
				'edges_in_graph:',
				self.in_graph.sum().item(),
			)

		elif level == 'edge':
			valid_edges = self.real_edge_mask

			info = self._rank_indices_1d(
				self.scores,
				valid_mask=valid_edges,
				n=n,
				absolute=absolute,
				include_zero_scores=include_zero_scores,
			)
			idxs = info["idxs"]

			flat_edges = self.in_graph.view(-1)
			flat_edges[:] = False
			flat_edges[idxs] = True

			if reset:
				nodes_with_outgoing = self.in_graph.any(dim=1)
				nodes_with_ingoing = einsum(
					self.in_graph.any(dim=0).float(),
					self.forward_to_backward.float(),
					'backward, forward backward -> forward',
				) > 0
				nodes_with_ingoing[0] = True
				self.nodes_in_graph |= nodes_with_outgoing & nodes_with_ingoing

			print('[dbg] edges_in_graph:', self.in_graph.sum().item())

		else:
			raise ValueError(f"Invalid level: {level}")

		if prune:
			self.prune()

	def prune(self):
		"""Converts a potentially messy Graph into one that is fully connected. The number of components after this is done is strictly non-increasing; it may remove nodes or edges from the graph, but it won't add them. This function first removes nodes with no neurons (if applicable). Then, it repeatedly removes nodes that lack incoming or outgoing edges (or both), and then edges missing a parent or child. Finally, it pruned the neurons of any removed nodes.
		"""

		assert (self.in_graph <= self.real_edge_mask).all(), "Found non-real edges turned on"
		
		# remove neuronless nodes
		if self.neurons_in_graph is not None:
			self.nodes_in_graph &= self.neurons_in_graph.any(dim=1)
		
		old_new_same = False
		# Could take twice as many iterations as there are layers! But will probably not
		while not old_new_same:
			# remove nodes with 0 incoming or outgoing edges
			nodes_with_outgoing = self.in_graph.any(dim=1)
			nodes_with_ingoing = einsum(self.in_graph.any(dim=0).float(), self.forward_to_backward.float(), 'backward, forward backward -> forward') > 0
			nodes_with_ingoing[0] = True  # input node always treated as if it has incoming edges
			
			old_nodes_in_graph = self.nodes_in_graph.clone()
			self.nodes_in_graph[:] = nodes_with_outgoing & nodes_with_ingoing
			
			# remove edges with missing parents or children
			forward_in_graph = self.nodes_in_graph.float()
			backward_in_graph = (self.nodes_in_graph.float() @ self.forward_to_backward.float())
			backward_in_graph[-1] = 1  # logits node is always present
			
			old_edges_in_graph = self.in_graph.clone()
			edge_remask = (
				self.real_edge_mask & 
				(einsum(forward_in_graph, backward_in_graph, 'forward, backward -> forward backward') > 0)
			)
			self.in_graph &= edge_remask

			old_new_same = (
				torch.all(old_nodes_in_graph == self.nodes_in_graph) and
				torch.all(old_edges_in_graph == self.in_graph)
			)
			
		# remove neurons from nodes not in the graph
		if self.neurons_in_graph is not None:
			self.neurons_in_graph &= self.nodes_in_graph.view(-1, 1)

		assert self.nodes_in_graph[0].item(), "Input got pruned unexpectedly"
		# assert self.nodes_in_graph[logits_i].item(), "Logits got pruned unexpectedly"
		assert (self.in_graph <= self.real_edge_mask).all(), "Phantom edges survived pruning"
			

	@classmethod
	def from_model(cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict], neuron_level: bool = False, node_scores: bool = False) -> 'Graph':
		"""Instantiate a Graph object from a HookedTransformer or HookedTransformerConfig object, or a similar Dict. The neuron_level parameter determines whether the graph should be neuron-level or not, while the node_scores parameter determines whether the graph should have node scores or not. If you don't have scores for all nodes / neurons, just don't set them (default is torch.nan). Any node/neuron without a real score will always be kept in the graph when doing node/neuron-level topn (but might be eliminated by another level's topn, e.g. a node with no neuron scores might be removed if it loses all edges)

		Args:
			model_or_config (Union[HookedTransformer,HookedTransformerConfig, Dict]): A config object; it needs to contain n_layers, n_heads, parallel_attn_mlp, and d_model.
			neuron_level (bool, optional): _description_. Defaults to False.
			node_scores (bool, optional): _description_. Defaults to False.

		Raises:
			ValueError: If you pass an invalid type for model_or_config

		Returns:
			_type_: a Graph
		"""
		graph = Graph()
		graph.cfg = GraphConfig()
		if isinstance(model_or_config, HookedTransformer):
			cfg = model_or_config.cfg
			graph.cfg.update({'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'd_model': cfg.d_model})
		elif isinstance(model_or_config, HookedTransformerConfig):
			cfg = model_or_config
			graph.cfg.update({'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'd_model': cfg.d_model})
		elif isinstance(model_or_config, dict):
			graph.cfg.update(model_or_config)
		else:
			raise ValueError(f"Invalid input type: {type(model_or_config)}")
			
		graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
		graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1
		graph.forward_to_backward = torch.zeros((graph.n_forward, graph.n_backward)).bool()
		
		graph.scores = torch.zeros((graph.n_forward, graph.n_backward))
		graph.real_edge_mask = torch.zeros((graph.n_forward, graph.n_backward)).bool()
		graph.in_graph = torch.zeros((graph.n_forward, graph.n_backward)).bool()
		graph.nodes_in_graph = torch.zeros(graph.n_forward).bool()
		if node_scores:
			graph.nodes_scores = torch.zeros(graph.n_forward) 
			graph.nodes_scores[:] = torch.nan
		else:
			graph.nodes_scores = None
		if neuron_level:
			graph.neurons_in_graph = torch.zeros((graph.n_forward, graph.cfg['d_model'])).bool()
			graph.neurons_scores = torch.zeros((graph.n_forward, graph.cfg['d_model']))
			graph.neurons_scores[:] = torch.nan
		else:
			graph.neurons_in_graph = None
			graph.neurons_scores = None
		
		input_node = InputNode(graph)
		graph.nodes[input_node.name] = input_node
		residual_stream = [input_node]

		for layer in range(graph.cfg['n_layers']):
			attn_nodes = [AttentionNode(layer, head, graph) for head in range(graph.cfg['n_heads'])]
			mlp_node = MLPNode(layer, graph)
			
			for attn_node in attn_nodes: 
				graph.nodes[attn_node.name] = attn_node 
				for letter in 'qkv':
					graph.forward_to_backward[graph.forward_index(attn_node, attn_slice=False), graph.backward_index(attn_node, attn_slice=False, qkv=letter)] = True
			graph.nodes[mlp_node.name] = mlp_node     
			graph.forward_to_backward[graph.forward_index(mlp_node, attn_slice=False), graph.backward_index(mlp_node, attn_slice=False)] = True
									
			if graph.cfg['parallel_attn_mlp']:
				for node in residual_stream:
					for attn_node in attn_nodes:          
						for letter in 'qkv':           
							graph.add_edge(node, attn_node, qkv=letter)
					graph.add_edge(node, mlp_node)
				
				residual_stream += attn_nodes
				residual_stream.append(mlp_node)

			else:
				for node in residual_stream:
					for attn_node in attn_nodes:     
						for letter in 'qkv':           
							graph.add_edge(node, attn_node, qkv=letter)
				residual_stream += attn_nodes

				for node in residual_stream:
					graph.add_edge(node, mlp_node)
				residual_stream.append(mlp_node)
						
		logit_node = LogitNode(graph.cfg['n_layers'], graph)
		for node in residual_stream:
			graph.add_edge(node, logit_node)
			
		graph.nodes[logit_node.name] = logit_node

		return graph


			
