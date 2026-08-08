# import os
# # Force single-threaded usage in BLAS/OpenBLAS/MKL/NumExpr
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from more_itertools import unique_everseen
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from functools import reduce
from tqdm import tqdm
import math

import numbers
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data, _check_sample_weight
from sklearn.linear_model import enet_path as _enet_path_base
from sklearn.linear_model._coordinate_descent import _pre_fit, _set_order
# from scipy import sparse

from scipy import sparse

from rulefit.rulefit import Winsorizer, FriedScale

from collections import defaultdict
from xgboost import XGBRegressor, XGBClassifier

from numba import njit
from numba import int32, float64, optional, types
from numba.experimental import jitclass
from numba.typed import List

@njit
def float_to_str(value, n_decimals=6):
	if value < 0:
		is_negative = True
		value = -value
	else:
		is_negative = False

	integer_part = int(value)
	decimal_part = value - integer_part

	# Convert integer part to string
	int_str = ""
	if integer_part == 0:
		int_str = "0"
	else:
		while integer_part > 0:
			int_str = chr(48 + integer_part % 10) + int_str
			integer_part //= 10

	# Convert decimal part to string
	dec_str = ""
	if decimal_part > 0:
		dec_str = "."
		count = 0
		while decimal_part > 0 and count < n_decimals:  # Limit decimal places
			decimal_part *= 10
			digit = int(decimal_part)
			dec_str += chr(48 + digit)
			decimal_part -= digit
			count += 1

	# Combine parts
	result = int_str + dec_str
	if is_negative:
		result = "-" + result

	return result

@njit(fastmath=True)
def rand_int(max_value, seed):
	"""
	Generate a random integer between 0 and max_value (exclusive).

	Parameters
	----------
	max_value : int
		The upper bound of the range (exclusive).
	rng_state : numpy.random.Generator
		The random number generator state (NumPy Generator instance).
		Must be seeded and passed externally for reproducibility.

	Returns
	-------
	int
		Random integer between 0 and max_value - 1.
	"""
	# Linear Congruential Generator (LCG) for simple random number generation
	seed = (1103515245 * seed + 12345) % (2**31)
	return seed % max_value

def shap_enet_path(
	X,
	y,
	*,
	l1_ratio=0.5,
	eps=1e-3,
	n_alphas=100,
	alphas=None,
	precompute="auto",
	Xy=None,
	copy_X=True,
	coef_init=None,
	verbose=False,
	return_n_iter=False,
	positive=False,
	check_input=True,
	shap_weights=None,
	quiet=False,
	**params,
):
	"""
	SHAP-weighted Elastic-Net path using sklearn's C-implemented solver.

	Equivalent (up to fp noise) to your numba solver that optimizes:

		1/2 ||y - X w||^2
		+ alpha * sum_j |w_j| / s_j
		+ (beta/2) * sum_j (w_j / s_j)^2

	via change of variables w_j = s_j * v_j, X' = X * s.
	"""

	# If no shap weights: behave exactly like plain enet_path
	if shap_weights is None:
		return _enet_path_base(
			X,
			y,
			l1_ratio=l1_ratio,
			eps=eps,
			n_alphas=n_alphas,
			alphas=alphas,
			precompute=precompute,
			Xy=Xy,
			copy_X=copy_X,
			coef_init=coef_init,
			verbose=verbose,
			return_n_iter=return_n_iter,
			positive=positive,
			check_input=check_input,
			**params,
		)

	# ---- validate shap_weights ----
	sw = np.asarray(shap_weights, dtype=float)
	if sw.ndim != 1:
		raise ValueError(
			f"shap_weights must be 1D of length n_features, got shape {sw.shape}"
		)
	if sw.shape[0] != X.shape[1]:
		raise ValueError(
			f"shap_weights has length {sw.shape[0]} but X has {X.shape[1]} features"
		)
	if np.any(sw <= 0):
		raise ValueError("shap_weights must be strictly positive for this transformation.")

	# ---- scale X columns: X' = X * diag(sw) ----
	if sparse.isspmatrix(X):
		X_scaled = X.multiply(sw)
	else:
		X_scaled = X * sw

	# ---- handle precompute / Gram matrix / Xy ----
	precompute_local = precompute
	Xy_local = None

	# If precompute is an actual Gram matrix (your error case), drop it.
	if (not isinstance(precompute, str)) and hasattr(precompute, "shape"):
		if precompute.shape[0] == precompute.shape[1]:
			# This is a Gram for *unscaled* X and will fail validation.
			# Let sklearn recompute everything from X_scaled.
			precompute_local = False
	# Otherwise, we can keep precompute as 'auto'/True/False
	# and scale Xy if it was provided.
	if Xy is not None and precompute_local is not False:
		Xy_arr = np.asarray(Xy, dtype=float)
		if Xy_arr.ndim == 1:
			Xy_local = Xy_arr * sw
		else:
			# (n_features, n_targets)
			Xy_local = Xy_arr * sw[:, None]

	# Clean up params so sklearn's enet_path doesn't see SHAP-specific stuff
	params = params.copy()
	params.pop("shap_weights", None)
	params.pop("quiet", None)

	# ---- call sklearn's C-backed enet_path on (X_scaled, y) ----
	result = _enet_path_base(
		X_scaled,
		y,
		l1_ratio=l1_ratio,
		eps=eps,
		n_alphas=n_alphas,
		alphas=alphas,
		precompute=precompute_local,
		Xy=Xy_local,
		copy_X=copy_X,
		coef_init=coef_init,
		verbose=verbose,
		return_n_iter=return_n_iter,
		positive=positive,
		check_input=check_input,
		**params,
	)

	if return_n_iter:
		alphas_out, coefs_v, dual_gaps, n_iters = result
	else:
		alphas_out, coefs_v, dual_gaps = result

	# ---- map v -> w = s ⊙ v along feature axis ----
	sw_cast = sw.astype(coefs_v.dtype, copy=False)
	if coefs_v.ndim == 2:
		# (n_features, n_alphas)
		coefs_w = coefs_v * sw_cast[:, None]
	else:
		# (n_targets, n_features, n_alphas)
		coefs_w = coefs_v * sw_cast[None, :, None]

	if return_n_iter:
		return alphas_out, coefs_w, dual_gaps, n_iters
	return alphas_out, coefs_w, dual_gaps


def lasso_path(
	X,
	y,
	*,
	eps=1e-3,
	n_alphas=100,
	alphas=None,
	precompute="auto",
	Xy=None,
	copy_X=True,
	coef_init=None,
	verbose=False,
	return_n_iter=False,
	positive=False,
	shap_weights=None,
	quiet=False,
	**params,
):
	"""
	SHAP-weighted Lasso path as a special case of shap_enet_path with l1_ratio=1.
	"""
	params['check_input'] = True # external callers normally want validation
	return shap_enet_path(
		X,
		y,
		l1_ratio=1.0,
		eps=eps,
		n_alphas=n_alphas,
		alphas=alphas,
		precompute=precompute,
		Xy=Xy,
		copy_X=copy_X,
		coef_init=coef_init,
		verbose=verbose,
		return_n_iter=return_n_iter,
		positive=positive,
		# check_input=True, 
		shap_weights=shap_weights,
		quiet=quiet,
		**params,
	)


class SHAPLasso(Lasso):

	# use the SHAP-aware lasso_path defined above
	path = staticmethod(lasso_path)

	def __init__(
		self,
		alpha=1.0,
		*,
		fit_intercept=True,
		precompute=False,
		copy_X=True,
		max_iter=1000,
		tol=1e-4,
		warm_start=False,
		positive=False,
		random_state=None,
		selection="cyclic",
		shap_weights=None,
		quiet=False,
	):
		super().__init__(
			alpha=alpha,
			fit_intercept=fit_intercept,
			precompute=precompute,
			copy_X=copy_X,
			max_iter=max_iter,
			tol=tol,
			warm_start=warm_start,
			positive=positive,
			random_state=random_state,
			selection=selection,
		)
		self.shap_weights = shap_weights
		self.quiet = quiet

	def fit(self, X, y, sample_weight=None, check_input=True):
		if self.alpha == 0:
			warnings.warn(
				(
					"With alpha=0, this algorithm does not converge "
					"well. You are advised to use the LinearRegression "
					"estimator"
				),
				stacklevel=2,
			)

		X_copied = False
		if check_input:
			X_copied = self.copy_X and self.fit_intercept
			X, y = validate_data(
				self,
				X,
				y,
				accept_sparse="csc",
				order="F",
				dtype=[np.float64, np.float32],
				force_writeable=True,
				accept_large_sparse=False,
				copy=X_copied,
				multi_output=True,
				y_numeric=True,
			)
			y = check_array(
				y, order="F", copy=False, dtype=X.dtype.type, ensure_2d=False
			)

		n_samples, n_features = X.shape
		alpha = self.alpha

		if isinstance(sample_weight, numbers.Number):
			sample_weight = None
		if sample_weight is not None:
			if check_input:
				sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
			sample_weight = sample_weight * (n_samples / np.sum(sample_weight))

		should_copy = self.copy_X and not X_copied
		X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
			X,
			y,
			None,
			self.precompute,
			fit_intercept=self.fit_intercept,
			copy=should_copy,
			check_input=check_input,
			sample_weight=sample_weight,
		)
		if check_input or sample_weight is not None:
			X, y = _set_order(X, y, order="F")
		if y.ndim == 1:
			y = y[:, np.newaxis]
		if Xy is not None and Xy.ndim == 1:
			Xy = Xy[:, np.newaxis]

		n_targets = y.shape[1]
		if not self.warm_start or not hasattr(self, "coef_"):
			coef_ = np.zeros((n_targets, n_features), dtype=X.dtype, order="F")
		else:
			coef_ = self.coef_
			if coef_.ndim == 1:
				coef_ = coef_[np.newaxis, :]

		dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
		self.n_iter_ = []

		n_targets_range = range(n_targets)
		if not self.quiet and n_targets > 1:
			n_targets_range = tqdm(n_targets_range, total=n_targets, desc="Target features processed")

		for k in n_targets_range:
			if Xy is not None:
				this_Xy = Xy[:, k]
			else:
				this_Xy = None

			_, this_coef, this_dual_gap, this_iter = self.path(
				X,
				y[:, k],
				eps=None,
				n_alphas=None,
				alphas=[alpha],
				precompute=precompute,
				Xy=this_Xy,
				copy_X=True,
				coef_init=coef_[k],
				verbose=False,
				return_n_iter=True,
				positive=self.positive,
				check_input=False,
				# forwarded params to enet_path
				tol=self.tol,
				X_offset=X_offset,
				X_scale=X_scale,
				max_iter=self.max_iter,
				random_state=self.random_state,
				selection=self.selection,
				sample_weight=sample_weight,
				shap_weights=self.shap_weights,
				quiet=self.quiet,
			)
			coef_[k] = this_coef[:, 0]
			dual_gaps_[k] = this_dual_gap[0]
			self.n_iter_.append(this_iter[0])

		if n_targets == 1:
			self.n_iter_ = self.n_iter_[0]
			self.coef_ = coef_[0]
			self.dual_gap_ = dual_gaps_[0]
		else:
			self.coef_ = coef_
			self.dual_gap_ = dual_gaps_

		self._set_intercept(X_offset, y_offset, X_scale)

		if not all(np.isfinite(w).all() for w in [self.coef_, self.intercept_]):
			raise ValueError(
				"Coordinate descent iterations resulted in non-finite parameter "
				"values. The input data may contain large values and need to "
				"be preprocessed."
			)

		return self


class SHAPLassoCV(LassoCV):

	# use the same SHAP-aware path
	path = staticmethod(lasso_path)

	def __init__(
		self,
		*,
		eps=1e-3,
		n_alphas=100,
		alphas=None,
		fit_intercept=True,
		precompute="auto",
		max_iter=1000,
		tol=1e-4,
		copy_X=True,
		cv=None,
		verbose=False,
		n_jobs=None,
		positive=False,
		random_state=None,
		selection="cyclic",
		shap_weights=None,
		quiet=False,
		**args,
	):
		super().__init__(
			eps=eps,
			n_alphas=n_alphas,
			alphas=alphas,
			fit_intercept=fit_intercept,
			precompute=precompute,
			max_iter=max_iter,
			tol=tol,
			copy_X=copy_X,
			cv=cv,
			verbose=verbose,
			n_jobs=n_jobs,
			positive=positive,
			random_state=random_state,
			selection=selection,
			**args,
		)
		self.shap_weights = shap_weights
		self.quiet = quiet

	def _get_estimator(self):
		# return SHAPLasso(shap_weights=self.shap_weights)
		return SHAPLasso(
			shap_weights=self.shap_weights,
			quiet=self.quiet,
			alpha=getattr(self, "alpha_", 1.0),
			fit_intercept=self.fit_intercept,
			precompute=self.precompute,
			copy_X=self.copy_X,
			max_iter=self.max_iter,
			tol=self.tol,
			warm_start=False,
			positive=self.positive,
			random_state=self.random_state,
			selection=self.selection,
		)

# class Winsorizer():
#   """Performs Winsorization 1->1*

#   Warning: this class should not be used directly.
#   """    
#   def __init__(self,trim_quantile=0.0):
#       self.trim_quantile=trim_quantile
#       self.winsor_lims=None
		
#   def train(self,X):
#       # get winsor limits
#       self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
#       self.winsor_lims[0,:]=-np.inf
#       if self.trim_quantile>0:
#           for i_col in np.arange(X.shape[1]):
#               lower=np.percentile(X[:,i_col],self.trim_quantile*100)
#               upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
#               self.winsor_lims[:,i_col]=[lower,upper]
		
#   def trim(self,X):
#       X_=X.copy()
#       X_=np.where(X>self.winsor_lims[1,:],np.tile(self.winsor_lims[1,:],[X.shape[0],1]),np.where(X<self.winsor_lims[0,:],np.tile(self.winsor_lims[0,:],[X.shape[0],1]),X))
#       return X_

# class FriedScale():
#   """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

#   Each variable is first Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
#   Warning: this class should not be used directly.
#   """    
#   def __init__(self, winsorizer = None):
#       self.scale_multipliers=None
#       self.winsorizer = winsorizer
		
#   def train(self,X):
#       # get multipliers
#       if self.winsorizer != None:
#           X_trimmed= self.winsorizer.trim(X)
#       else:
#           X_trimmed = X

#       scale_multipliers=np.ones(X.shape[1])
#       for i_col in np.arange(X.shape[1]):
#           num_uniq_vals=len(np.unique(X[:,i_col]))
#           if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
#               scale_multipliers[i_col]=0.4/(1.0e-12 + np.std(X_trimmed[:,i_col]))
#       self.scale_multipliers=scale_multipliers
		
#   def scale(self,X):
#       if self.winsorizer != None:
#           return self.winsorizer.trim(X)*self.scale_multipliers
#       else:
#           return X*self.scale_multipliers

# Define the data types for Numba jitclass
type_spec_rule_condition = [
	('feature_index', int32),
	('threshold', float64),
	('operator', types.unicode_type),
	('support', optional(float64)),
	('gain', optional(float64)),
	('feature_name', optional(types.unicode_type))
]

@jitclass(type_spec_rule_condition)
class RuleCondition:
	"""Class for binary rule condition."""

	def __init__(self, feature_index, threshold, operator, support=0.0, gain=0.0, feature_name=''):
		self.feature_index = feature_index
		self.threshold = threshold
		self.operator = operator
		self.support = support
		self.gain = gain
		self.feature_name = feature_name

	def transform_weight(self, weight):
		return weight[self.feature_index]

	def transform(self, X):
		"""Transform dataset.

		Parameters
		----------
		X: array-like matrix, shape=(n_samples, n_features)

		Returns
		-------
		X_transformed: array-like matrix, shape=(n_samples, 1)
		"""
		if self.operator == "<=":
			res = 1 * (X[:, self.feature_index] <= self.threshold)
		elif self.operator == ">":
			res = 1 * (X[:, self.feature_index] > self.threshold)
		elif self.operator == ">=":
			res = 1 * (X[:, self.feature_index] >= self.threshold)
		elif self.operator == "<":
			res = 1 * (X[:, self.feature_index] < self.threshold)
		else:
			raise ValueError("Unsupported operator")
		return res

	def get_unique_key(self):
		return (self.threshold, self.operator, self.feature_name)

	def __hash__(self):
		return hash(self.get_unique_key())

	def __str__(self):
		feature = self.feature_name if self.feature_name else f"f{self.feature_index}"
		# Use Numba-compatible formatting for the threshold
		threshold_str = float_to_str(self.threshold, 2)  # Format float as a concise string
		# Combine strings into the final representation
		return feature + " " + self.operator + " " + threshold_str

# # Make sure RuleCondition is compiled and ready
# _ = RuleCondition.class_type.instance_type

# Define the data types for Numba jitclass
type_spec_rule = [
	('conditions', types.ListType(RuleCondition.class_type.instance_type)),
	('prediction_value', optional(float64)),
	('is_negated', optional(types.boolean)),
	('max_gain', float64),
	('total_gain', float64),
	('min_rule_coverage', float64),
	('global_coverage', float64),
]

@jitclass(type_spec_rule)
class Rule():
	"""Class for binary Rules from list of conditions"""

	def __init__(self, rule_conditions, prediction_value=0., is_negated=False, global_coverage=-1):
		# Convert the input list to a numba.typed.List
		self.conditions = rule_conditions
		self.conditions.sort(key=lambda x: (x.feature_index, x.threshold))
		self.max_gain = np.max(np.array([x.gain for x in rule_conditions]))
		self.total_gain = sum([x.gain for x in rule_conditions])
		self.min_rule_coverage = np.min(np.array([x.support for x in rule_conditions]))
		# self.total_rule_coverage = sum((x.support for x in rule_conditions))
		self.prediction_value = prediction_value
		self.is_negated = is_negated
		self.global_coverage = global_coverage

	def transform_weight(self, weight): ### My edit
		unique_conditions = list({c.feature_index: c for c in self.conditions}.values())  # Avoid duplicates by feature_index
		# if len(self.conditions) != len(unique_conditions):
		# 	print(len(self.conditions), len(unique_conditions))
		weight_applies = [condition.transform_weight(weight) for condition in unique_conditions]
		return reduce(lambda x,y: x + y, weight_applies)/len(weight_applies) # Prioritize rules with fewer more important conditions/features over rules with many less important conditions/features

	def transform(self, X):
		rule_applies = [condition.transform(X) for condition in self.conditions]
		if not self.is_negated:
			return reduce(lambda x,y: x * y, rule_applies)
		return reduce(lambda x,y: np.maximum(x,y), rule_applies)

	def get_coverage(self, X):
		return np.mean(self.transform(X))

	def __str__(self):
		if not self.is_negated:
			return  " & ".join([str(x) for x in self.conditions])
		return  " | ".join([str(x) for x in self.conditions])

	def __hash__(self):
		return sum([hash(condition) for condition in self.conditions])

	def __gt__(self, other):
		if self.is_negated and other.is_negated:
			return len(set([hash(cond) for cond in other.conditions]) - set([hash(cond) for cond in self.conditions])) == 0
		if not self.is_negated and not other.is_negated:
			return len(set([hash(cond) for cond in self.conditions]) - set([hash(cond) for cond in other.conditions])) == 0
		return False

	def __lt__(self, other):
		if self.is_negated and other.is_negated:
			return len(set([hash(cond) for cond in self.conditions]) - set([hash(cond) for cond in other.conditions])) == 0
		if not self.is_negated and not other.is_negated:
			return len(set([hash(cond) for cond in other.conditions]) - set([hash(cond) for cond in self.conditions])) == 0
		return False

def negate_operator(operator):
	if operator == '>=':
		return '<'
	if operator == '<=':
		return '>'
	if operator == '>':
		return '<='
	if operator == '<':
		return '>='
	raise ValueError("Unsupported operator")

def negate_rule(rule):
	conditions = [
		RuleCondition(cond.feature_index, cond.threshold, negate_operator(cond.operator), support=cond.support, gain=cond.gain, feature_name=cond.feature_name)
		for cond in rule.conditions
	]
	negated_rule = Rule(List(conditions), prediction_value=rule.prediction_value, is_negated=not rule.is_negated, global_coverage=1-rule.global_coverage)
	return negated_rule

class RuleEnsemble():
	"""Ensemble of binary decision rules

	This class implements an ensemble of decision rules that extracts rules from
	an ensemble of decision trees.

	Parameters
	----------
	tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
		Trees from which the rules are created

	feature_names: List of strings, optional (default=None)
		Names of the features

	Attributes
	----------
	rules: List of Rule
		The ensemble of rules extracted from the trees
	"""
	def __init__(self, tree_list=None, tree_dump=None, datapoints=None, feature_names=None):
		self.feature_names = feature_names
		self.datapoints_count = len(datapoints)
		
		if tree_dump:
			self.rules = list(unique_everseen((
				rule
				for tree_dump in map(json.loads, tree_dump)
				for rule in RuleEnsemble.extract_rules_from_tree_dump(tree_dump, self.datapoints_count, feature_names=self.feature_names)
			), key=hash))
		else:
			self.rules = list(unique_everseen((
				rule
				for tree in tree_list
				for rule in RuleEnsemble.extract_rules_from_tree_list(tree[0].tree_, self.datapoints_count, feature_names=self.feature_names)
			), key=hash))
		for rule in self.rules:
			rule.global_coverage = rule.get_coverage(datapoints)
		self.rules = list(filter(lambda x: x.global_coverage != 0, self.rules)) # Remove any rule not covered by real data
		# self.rules += list(map(negate_rule, self.rules))
		print("Rules found:", len(self.rules))

	@staticmethod
	def extract_rules_from_tree_list(tree, datapoints_count, feature_names=None):
		"""Helper to turn a tree into as set of rules
		"""
		rules = set()

		def traverse_nodes(node_id=0, operator=None, threshold=None, feature=None, conditions=[]):
			# Children
			left_child = tree.children_left[node_id]
			right_child = tree.children_right[node_id]
			
			if node_id != 0:
				if feature_names is not None:
					feature_name = feature_names[feature]
				else:
					feature_name = feature

				# Impurities
				parent_impurity = tree.impurity[node_id]
				left_impurity = tree.impurity[left_child]
				right_impurity = tree.impurity[right_child]

				# Number of samples
				parent_samples = tree.weighted_n_node_samples[node_id]
				left_samples = tree.weighted_n_node_samples[left_child]
				right_samples = tree.weighted_n_node_samples[right_child]

				# Weighted impurity of children
				weighted_impurity = (
					(left_samples * left_impurity + right_samples * right_impurity) / parent_samples
				)
				
				# Gain
				gain = parent_impurity - weighted_impurity

				rule_condition = RuleCondition(feature_index=feature,
											   threshold=threshold,
											   operator=operator,
											   support = tree.n_node_samples[node_id] / datapoints_count,
											   gain=gain,
											   feature_name=feature_name)
				new_conditions = conditions + [rule_condition]
			else:
				new_conditions = []
			## if not terminal node
			if left_child != right_child: 
				feature = tree.feature[node_id]
				threshold = tree.threshold[node_id]
				traverse_nodes(left_child, "<=", threshold, feature, new_conditions)
				traverse_nodes(right_child, ">", threshold, feature, new_conditions)
			else: # a leaf node
				if len(new_conditions)>0:
					new_rule = Rule(List(RuleEnsemble.filter_conditions(new_conditions)),tree.value[node_id][0][0])
					rules.update([new_rule])
				else:
					pass #tree only has a root node!
				return None

		traverse_nodes()
		
		return rules

	@staticmethod
	def extract_rules_from_tree_dump(tree, datapoints_count, feature_names=None):
		"""Recursively extract rules from a JSON XGBoost tree."""
		rules = []

		# print('tree:', json.dumps(tree, indent=4))

		def traverse_nodes(node, conditions=[]):
			# Base case: Check if it's a leaf node
			if 'leaf' in node:
				if conditions:
					new_rule = Rule(List(RuleEnsemble.filter_conditions(conditions)), node['leaf'])
					rules.append(new_rule)
				return

			# Extract split details
			feature = node['split']
			feature_index = int(feature[1:]) if feature.startswith('f') else int(feature)
			feature_name = feature_names[feature_index] if feature_names else feature
			threshold = node['split_condition']
			support = node['cover'] / datapoints_count
			gain = node['gain']
			yes_child = node['yes']
			no_child = node['no']
			missing_child = node['missing']

			# Condition for the left child (<= threshold)
			left_condition = RuleCondition(feature_index=feature_index,
											 threshold=threshold,
											 operator="<=",
											 support=support,
											 gain=gain,
											 feature_name=feature_name)

			# Condition for the right child (> threshold)
			right_condition = RuleCondition(feature_index=feature_index,
											threshold=threshold,
											operator=">",
											support=support,
											gain=gain,
											feature_name=feature_name)

			# Recurse for children with updated conditions
			for child in node.get('children', []):
				if child['nodeid'] == yes_child:  # Left child
					traverse_nodes(child, conditions + [left_condition])
				elif child['nodeid'] == no_child:  # Right child
					traverse_nodes(child, conditions + [right_condition])

		traverse_nodes(tree)
		return rules

	@staticmethod
	def filter_conditions(conditions):
		"""
		Filters RuleCondition objects:
		- Keeps only the highest threshold for operator '>='
		- Keeps only the lowest threshold for operator '<='

		Parameters:
		-----------
		conditions : list[RuleCondition]
			List of RuleCondition objects to filter.

		Returns:
		--------
		filtered_conditions : list[RuleCondition]
			Filtered list of RuleCondition objects.
		"""

		grouped_conditions = defaultdict(list)

		# Group conditions by feature_index and operator
		for condition in conditions:
			key = (condition.feature_index, condition.operator.strip('='))
			grouped_conditions[key].append(condition)

		filtered_conditions = []

		# Apply filtering logic
		for (feature_index, operator), group in grouped_conditions.items():
			if operator == '>':
				# Keep condition with the highest threshold
				best_condition = max(group, key=lambda x: (x.threshold, 0 if x.operator.endswith('=') else 1))
			elif operator == '<':
				# Keep condition with the lowest threshold
				best_condition = min(group, key=lambda x: (x.threshold, 1 if x.operator.endswith('=') else 0))
			
			filtered_conditions.append(best_condition)

		return filtered_conditions

	def transform(self, X, shap_weights=None):
		"""Transform dataset.

		Parameters
		----------
		X:      array-like matrix, shape=(n_samples, n_features)
		Returns
		-------
		X_transformed: array-like matrix, shape=(n_samples, n_out)
			Transformed dataset. Each column represents one rule.
		"""
		transformed_shap_weights = np.array([rule.transform_weight(shap_weights) for rule in self.rules]).T if shap_weights is not None else None ### My edit
		return np.array([rule.transform(X) for rule in self.rules]).T, transformed_shap_weights

	def __str__(self):
		return ' '.join(map(str, self.rules))

class RuleSHAP(BaseEstimator, TransformerMixin):

	def __init__(self, gboost_config_dict=None, model_type='rl', rfmode='regress', lin_trim_quantile=0.025, lin_standardise=True, Cs=None, cv=3, random_state=None, max_rules=4000, tree_size=10):
		if gboost_config_dict is None:
			gboost_config_dict = {
				'n_estimators': 100, # Number of boosting rounds. Each round is a tree
				'max_depth': 10, # max rule lenght
				'subsample': 0.8, # Subsample ratio of the training instance.
				# 'max_leaves': 50, # Maximum number of terminal nodes (leaves)
				# 'learning_rate': 0.01,
			}
		gboost_config_dict['random_state'] = random_state
		self.gboost_config_dict = gboost_config_dict
		self.model_type = model_type
		self.rfmode = rfmode
		self.lin_trim_quantile=lin_trim_quantile
		self.lin_standardise=lin_standardise
		self.winsorizer=Winsorizer(trim_quantile=lin_trim_quantile)
		self.friedscale=FriedScale(self.winsorizer)
		self.Cs = Cs
		self.cv = cv
		self.random_state = random_state
		self.max_rules = max_rules
		self.tree_size = tree_size
	
	def fit(self, X, y=None, feature_names=None, sample_weight=1, shap_weights=None, use_shap_in_xgb=True, use_shap_in_lasso=True, compute_sparsity_coef=True):

		print('Calling RuleSHAP')
		self.n_features_in_ = X.shape[1]
		self.compute_sparsity_coef_ = bool(compute_sparsity_coef)

		if isinstance(shap_weights, (list, tuple)):
			shap_weights = np.array(shap_weights)

		# ensure no value is exactly 0
		shap_weights = np.where(shap_weights == 0, 1e-8, shap_weights)

		if isinstance(shap_weights, np.ndarray):
			assert len(shap_weights) == X.shape[1], "Feature weights must match the number of features!"
			assert np.all(shap_weights >= 0), "Feature weights must be non-negative!"
			assert np.all(np.isfinite(shap_weights)), "Feature weights must be finite numbers!"
			shap_weights = shap_weights/np.sum(shap_weights) # SHAP weights normalized in (0,1]

		N = X.shape[0]
		if feature_names is None:
			self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
		else:
			self.feature_names = feature_names

		if 'r' in self.model_type:
			# initialise tree generator
			if use_shap_in_xgb:
				feature_weights = shap_weights # Weight for each feature, defines the probability of each feature being selected when colsample is being used. All values must be greater than 0, otherwise a ValueError is thrown.
				self.gboost_config_dict['tree_method'] = 'exact'
				# self.gboost_config_dict['colsample_bytree'] = 1/len(self.feature_names) # Fraction of features considered for each tree. Limits the number of features, simplifying the rules. Use values around 0.5–0.8.
				self.gboost_config_dict['colsample_bylevel'] = 1/len(self.feature_names) # Fraction of features considered for each level. Limits the number of features, simplifying the rules.
			else:
				feature_weights = None

			y_arr = np.asarray(y).reshape(-1).astype(np.float64)
			self._y_train = y_arr
			self._y_train_bin = (y_arr >= 0.5).astype(np.int8)

			uniq = np.unique(self._y_train)
			is_binary = np.all(np.isin(uniq, [0.0, 1.0]))
			self._problem = 'classification' if (self.rfmode == 'classify' or is_binary) else 'regression'

			if self.rfmode == 'regress':
				self.tree_generator = XGBRegressor(**self.gboost_config_dict)
			else:
				self.tree_generator = XGBClassifier(**self.gboost_config_dict)

			self.tree_generator.fit(X, y, feature_weights=feature_weights)
			tree_dump = self.tree_generator.get_booster().get_dump(with_stats=True, dump_format='json') # Access the Booster and get individual trees as strings. Dump the individual trees into a list
			self.rule_ensemble = RuleEnsemble(datapoints=X, tree_dump=tree_dump, feature_names=self.feature_names) ## extract rules
			#############################################################################

			X_rules, rules_shap_weights = self.rule_ensemble.transform(X, shap_weights=shap_weights)

			# --- store training activations & labels for per-rule metrics ---
			self._X_train = X
			self._Z_train = X_rules.astype(np.uint8, copy=False)  # [n_samples, n_rules]

		# Stage-7 calls RuleSHAP with compute_sparsity_coef=False.  In that mode
		# downstream code only needs the tree rules and cached rule activations;
		# building the dense linear/LASSO design matrix is pure memory overhead.
		if not self.compute_sparsity_coef_:
			self.lscv = None
			self.coef_ = None
			self.intercept_ = None
			y_arr = np.asarray(y).reshape(-1).astype(np.float64)
			if np.isscalar(sample_weight):
				sw = np.full_like(y_arr, float(sample_weight), dtype=np.float64)
			else:
				sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
			self._y_mean_ = float(np.average(y_arr, weights=sw))
			return self

		# standardise linear variables if requested (for regression model only)
		if 'l' in self.model_type:
			self.winsorizer.train(X)
			winsorized_X = self.winsorizer.trim(X)
			self.stddev = np.std(winsorized_X, axis=0)
			self.mean = np.mean(winsorized_X, axis=0)

			if self.lin_standardise:
				self.friedscale.train(X)
				X_regn = self.friedscale.scale(X)
			else:
				X_regn = X.copy()

			regn_shap_weights = shap_weights

		# Compile Training data
		X_concat = np.zeros([X.shape[0], 0])
		concat_shap_weights = np.array([])

		if 'l' in self.model_type:
			X_concat = np.concatenate((X_concat, X_regn), axis=1)
			if shap_weights is not None:
				concat_shap_weights = np.concatenate((concat_shap_weights, regn_shap_weights))

		if 'r' in self.model_type:
			if X_rules.shape[0] > 0:
				X_concat = np.concatenate((X_concat, X_rules), axis=1)
				if shap_weights is not None:
					concat_shap_weights = np.concatenate((concat_shap_weights, rules_shap_weights))

		# fit Lasso (unchanged)
		if self.Cs is None:
			n_alphas = 100
			alphas = None
		elif hasattr(self.Cs, "__len__"):
			n_alphas = None
			alphas = 1. / self.Cs
		else:
			n_alphas= self.Cs
			alphas=None
		# Derive feature weights from SHAP values by using the mean absolute SHAP value per feature as its weight. This makes the shap_weights a probability distribution
		# concat_shap_weights = concat_shap_weights/np.max(concat_shap_weights)
		self.lscv = SHAPLassoCV(
			n_alphas=n_alphas,
			alphas=alphas,
			cv=self.cv,
			random_state=self.random_state,
			shap_weights=None if (not use_shap_in_lasso or concat_shap_weights.size == 0) else concat_shap_weights
		)
		self.lscv.fit(X_concat, y, sample_weight=sample_weight)
		self.coef_ = self.lscv.coef_
		self.intercept_ = self.lscv.intercept_
		return self

	def predict(self, X):
		# ---- if LASSO was skipped, fall back safely ----
		if getattr(self, "lscv", None) is None:
			if hasattr(self, "tree_generator"):
				return self.tree_generator.predict(X)
			# linear-only (or no model): constant baseline
			return np.full(X.shape[0], getattr(self, "_y_mean_", 0.0), dtype=np.float64)

		X_concat = np.zeros([X.shape[0], 0])
		if 'l' in self.model_type:
			if self.lin_standardise:
				X_concat = np.concatenate((X_concat, self.friedscale.scale(X)), axis=1)
			else:
				X_concat = np.concatenate((X_concat, X), axis=1)

		if 'r' in self.model_type:
			rule_coefs = self.coef_[-len(self.rule_ensemble.rules):]
			if len(rule_coefs) > 0:
				X_rules, _ = self.rule_ensemble.transform(X)
				if X_rules.shape[0] > 0:
					X_concat = np.concatenate((X_concat, X_rules), axis=1)

		return self.lscv.predict(X_concat)

	def transform(self, X=None, shap_weights=None):
		"""Transform dataset.

		Parameters
		----------
		X : array-like matrix, shape=(n_samples, n_features)
		  Input data to be transformed. Use ``dtype=np.float32`` for maximum
		  efficiency.

		Returns
		-------
		X_transformed: matrix, shape=(n_samples, n_out)
		  Transformed data set
		"""
		return self.rule_ensemble.transform(X, shap_weights=shap_weights)

	def get_rules(self, X=None, y=None, filter_out_empty_coef=True):
		"""
		If X is provided, compute per-rule metrics on that dataset.
		Otherwise, use the training data cached during fit().

		NOTE: metrics are computed against ŷ_model (self.predict), not y.
		"""

		# ---- detect whether LASSO ran ----
		has_lasso = (getattr(self, "lscv", None) is not None) and (getattr(self, "coef_", None) is not None)

		# rule_ensemble may not exist if model_type doesn't include 'r'
		if not hasattr(self, "rule_ensemble"):
			# No rules to report
			return pd.DataFrame(columns=[
				"rule_index","is_negated","rule_expression","component_type","dataset_coverage",
				"coefficient_sign","importance","min_condition_coverage","path_gain_sum","path_gain_max",
				"importance_weighted_by_gain"
			])

		assert has_lasso or not filter_out_empty_coef, "No coefficients exist; run LASSO regression first"

		output_rows = []

		have_cache = all(hasattr(self, a) for a in ("_Z_train", "_X_train"))
		have_external = (X is not None)

		if have_external:
			Z, _ = self.rule_ensemble.transform(X)
			Z = (Z > 0).astype(np.int8)
			X_mat = np.asarray(X)
			y_mat = np.asarray(y)
		else:
			Z = self._Z_train if have_cache else None
			X_mat = self._X_train if have_cache else None
			y_mat = self._y_train if have_cache else None

		have_data = (Z is not None) and (X_mat is not None) and (y_mat is not None)
		is_clf = getattr(self, "_problem", "regression") == "classification"

		eps = 1e-12
		lin_fire_q = 0.75

		if have_data:
			n, r = Z.shape
			sums = Z.sum(axis=0).astype(np.float64)
			not_sums = (n - sums).astype(np.float64)

			yhat = np.asarray(y_mat).reshape(-1)
			# yhat = np.asarray(self.predict(X_mat)).reshape(-1)
			if is_clf:
				if yhat.dtype.kind in "fc":
					yhatb = (yhat >= 0.5).astype(np.int8)
				else:
					yhatb = yhat.astype(np.int8)
			else:
				yhat_cont = yhat.astype(np.float64)
		else:
			n = r = 0
			sums = not_sums = None
			yhatb = None
			yhat_cont = None

		# ---- Linear effects ----
		# Only available if LASSO was fit
		if has_lasso:
			n_features = len(self.coef_) - len(self.rule_ensemble.rules)
		else:
			n_features = 0  # and we skip the linear section entirely

		if has_lasso:
			for i in range(n_features):
				coef = self.coef_[i]
				if self.lin_standardise:
					coef *= self.friedscale.scale_multipliers[i]
				if not coef:
					continue

				sign_pos = (coef > 0)
				impact_dir = "positive" if sign_pos else "negative"
				importance = abs(coef)

				row = [
					None, False,   # rule_index, is_negated
					self.feature_names[i], "linear",
					1.0,
					impact_dir, round(importance, 2),
					None, None, None, None,
				]

				if have_data:
					xi = X_mat[:, i].astype(np.float64)
					contr = np.abs(coef * xi)
					finite = np.isfinite(contr)
					if finite.any():
						tau = float(np.quantile(contr[finite], lin_fire_q))
						fire = finite & (contr >= tau)
					else:
						fire = np.zeros(n, dtype=bool)

					row[4] = round(float(fire.mean()), 2)

					if is_clf:
						row += [
							round(v, 2) if np.isfinite(v) else np.nan
							for v in self._signed_rule_metrics_against_target(fire, yhatb, sign_pos, eps=eps)
						]
					else:
						row += [
							float(v) if np.isfinite(v) else np.nan
							for v in self._segmentation_stats(fire, yhat_cont, sign_pos, tail_q=0.90, eps=eps)
						]

				output_rows.append(row)

		# ---- Rules ----
		rule_list = list(self.rule_ensemble.rules)
		if has_lasso:
			coef_list = self.coef_[n_features:]
		else:
			# No LASSO => no rule coefficients. Keep structure; coef=0 used only for display.
			coef_list = np.zeros(len(rule_list), dtype=np.float64)

		for j, (rule, coef) in enumerate(zip(rule_list, coef_list)):
			if filter_out_empty_coef and not coef:
				continue

			if have_data and n > 0:
				fire_base = (Z[:, j] == 1)

				if fire_base.any() and (~fire_base).any():
					if is_clf:
						if yhat.dtype.kind in "fc":
							score = float(yhat[fire_base].mean() - yhat[~fire_base].mean())
						else:
							score = float(yhatb[fire_base].mean() - yhatb[~fire_base].mean())
					else:
						score = float(yhat_cont[fire_base].mean() - yhat_cont[~fire_base].mean())
				else:
					score = np.sign(coef) if coef != 0 else 0.0

				if abs(score) < 1e-8 and coef != 0:
					score = np.sign(coef)

				effect_pos = (score >= 0)

				if effect_pos:
					rule_expr = str(rule)
					fire = fire_base
					dataset_cov = float(sums[j] / n)
					is_negated = False
				else:
					rule_expr = f"NOT({rule})"
					fire = ~fire_base
					dataset_cov = float(not_sums[j] / n)
					is_negated = True

				sign_pos_for_metrics = True
			else:
				score = coef
				effect_pos = (coef > 0)
				if effect_pos:
					rule_expr = str(rule)
					dataset_cov = float(rule.global_coverage)
					is_negated = False
				else:
					rule_expr = f"NOT({rule})"
					dataset_cov = float(1.0 - rule.global_coverage)
					is_negated = True

				fire = None
				sign_pos_for_metrics = True

			if has_lasso:
				importance = abs(coef)
			else:
				importance = abs(score)  # or MCC, depending on your preference
			weighted_importance = (1 + importance) * np.maximum(
				1, np.log(np.abs(rule.max_gain))
			)


			impact_dir = "positive"

			row = [
				int(j), 
				bool(is_negated),
				rule_expr,
				"rule",
				round(dataset_cov, 2),
				impact_dir,
				round(importance, 2),
				round(rule.min_rule_coverage, 2),
				round(rule.total_gain, 2),
				round(rule.max_gain, 2),
				round(float(weighted_importance), 2),
			]

			if have_data:
				if is_clf:
					row += [
						round(v, 2) if np.isfinite(v) else np.nan
						for v in self._signed_rule_metrics_against_target(
							fire, yhatb, sign_pos_for_metrics, eps=eps
						)
					]
				else:
					row += [
						float(v) if np.isfinite(v) else np.nan
						for v in self._segmentation_stats(
							fire, yhat_cont, sign_pos_for_metrics,
							tail_q=0.90, eps=eps
						)
					]

			output_rows.append(row)

		# ---- Columns ----
		cols = [
			"rule_index",
			"is_negated",
			"rule_expression",
			"component_type",
			"dataset_coverage",
			"coefficient_sign",
			"importance",
			"min_condition_coverage",
			"path_gain_sum",
			"path_gain_max",
			"importance_weighted_by_gain",
		]

		if have_data:
			if is_clf:
				cols += [
					"P(ŷ_model=coefficient_sign | fire)",
					"R(fire | ŷ_model=coefficient_sign)",
					"F1(ŷ_model=coefficient_sign | fire)",
					"Lift(ŷ_model=coefficient_sign | fire)",
					"Acc(ŷ_rule vs ŷ_model)",
					"Coverage(ŷ_model=1 correct by ŷ_rule)",  # TPR
					"Coverage(ŷ_model=0 correct by ŷ_rule)",  # TNR
					"BalancedAcc(ŷ_rule vs ŷ_model)",
					"MCC(ŷ_rule vs ŷ_model)",
				]
			else:
				cols += [
					"r(point-biserial; fire vs ŷ_model)",
					"Cohen_d(ŷ_model; fire vs ~fire)",
					"E[ŷ_model | fire] - E[ŷ_model | ~fire]",
					"R^2(ŷ_model segmented by fire)",
					"P(ŷ_model in signed_tail(q=0.90) | fire)",
					"R(ŷ_model in signed_tail(q=0.90) | fire)",
					"F1(ŷ_model in signed_tail(q=0.90) | fire)",
					"Lift(ŷ_model in signed_tail(q=0.90) | fire)",
				]

		rules_df = pd.DataFrame(output_rows, columns=cols)
		if is_clf:
			rules_df = rules_df.sort_values("MCC(ŷ_rule vs ŷ_model)", ascending=False)
		else:
			rules_df = rules_df.sort_values("importance_weighted_by_gain", ascending=False)
		return rules_df

	# ----------------------------
	# Fast OR optimizer via bitsets (objective: maximize MCC)
	# - Classification: MCC vs (y_target or self.predict)
	# - Regression: MCC vs a *tail event* derived from a continuous target
	#   (and also returns regression segmentation stats vs the continuous target)
	# ----------------------------
	def find_best_or_combo(
		self,
		rules_df,
		X=None,
		y_target=None,
		threshold=0.5,
		max_selected=None,
		return_pred=False,
		show_progress=True,
		topk_screen=None,          # keep only top-K literals by single-literal MCC
		min_single_mcc=-np.inf,    # optionally drop terrible literals early
		# (used when problem != classification):
		tail_q=0.90,               # tail quantile for event definition in regression
		sign_is_positive=True,      # True: top tail is event; False: bottom tail is event
		# --- Variant A: multi-seed greedy OR (still optimizes MCC) ---
		greedy_seed_metrics=None,             # e.g. ['MCC','F1','Lift','Coverage','Importance','WeightedImportance']
		greedy_seed_metric_directions=None,   # e.g. {'Coverage':'desc', 'FPR':'asc'}
		greedy_seed_topk=1,                   # take top-k seeds per metric
		greedy_seed_pool_k=None,              # if set, union top-k literals per metric into the screened pool
		return_greedy_runs=False,             # if True, include per-metric greedy summary in output
	):
		import math
		import numpy as np

		is_clf = getattr(self, "_problem", "regression") == "classification"

		if not hasattr(self, "rule_ensemble") or len(getattr(self.rule_ensemble, "rules", [])) == 0:
			return {"found_solution": False, "success": False,
					"reason": "No rules available (rule_ensemble missing or empty)."}

		# ---- Build / load Z ----
		have_cache = all(hasattr(self, a) for a in ("_Z_train", "_X_train"))
		if X is None:
			if not have_cache:
				return {"found_solution": False, "success": False,
						"reason": "No X provided and no cached training data."}
			Z = np.asarray(self._Z_train, dtype=np.int8)
			X_mat = np.asarray(self._X_train)
		else:
			X_mat = np.asarray(X)
			Z_rules, _ = self.rule_ensemble.transform(X_mat)
			Z = (Z_rules > 0).astype(np.int8)

		n, r = Z.shape
		r_rules = len(self.rule_ensemble.rules)
		if r == 0 or r_rules == 0:
			return {"found_solution": False, "success": False, "reason": "No rules available."}
		if r != r_rules:
			return {"found_solution": False, "success": False,
					"reason": f"Z has {r} columns but rule_ensemble has {r_rules} rules."}

		# ---- Build binary target yb ----
		# Classification: yb is provided y_target (binarized) or self.predict(X)
		# Regression: yb is a tail-event built from a continuous target (provided or self.predict(X))
		y_cont = None
		if is_clf:
			if y_target is None:
				yhat = np.asarray(self.predict(X_mat)).reshape(-1)
				if yhat.dtype.kind in "fc":
					yb = (yhat >= 0.5).astype(np.int8)
				else:
					yb = yhat.astype(np.int8)
				target_name = "yhat_model"
			else:
				target_raw = np.asarray(y_target).reshape(-1)
				if target_raw.size != n:
					return {"found_solution": False, "success": False,
							"reason": f"Target length mismatch: got {target_raw.size}, expected {n}."}
				if target_raw.dtype.kind in "fc":
					yb = (target_raw >= threshold).astype(np.int8)
				else:
					yb = target_raw.astype(np.int8)
				target_name = "y_target"
			evt_threshold = None
			evt_direction = None
		else:
			# continuous target
			if y_target is None:
				y_cont = np.asarray(self.predict(X_mat)).reshape(-1).astype(np.float64)
				target_name = "yhat_model_tail_event"
			else:
				y_cont = np.asarray(y_target).reshape(-1).astype(np.float64)
				if y_cont.size != n:
					return {"found_solution": False, "success": False,
							"reason": f"Target length mismatch: got {y_cont.size}, expected {n}."}
				target_name = "y_target_tail_event"

			# tail event
			if not (0.0 < float(tail_q) < 1.0):
				return {"found_solution": False, "success": False,
						"reason": f"tail_q must be in (0,1); got {tail_q}."}

			t_hi = float(np.nanquantile(y_cont, tail_q))
			t_lo = float(np.nanquantile(y_cont, 1.0 - tail_q))
			if sign_is_positive:
				yb = (y_cont >= t_hi).astype(np.int8)
				evt_threshold = t_hi
				evt_direction = f">= quantile({tail_q})"
			else:
				yb = (y_cont <= t_lo).astype(np.int8)
				evt_threshold = t_lo
				evt_direction = f"<= quantile({1.0 - tail_q})"

		y_pos = (yb == 1)
		P = int(y_pos.sum())
		N = int(n - P)

		# If event is degenerate, MCC is not informative; still return something stable.
		if P == 0 or N == 0:
			return {
				"found_solution": False,
				"success": False,
				"reason": f"Binary target is degenerate (P={P}, N={N}). "
						  f"{'Consider a different threshold.' if is_clf else 'Try a less extreme tail_q.'}",
				"target": target_name,
				"problem": "classification" if is_clf else "regression",
			}

		# ---- Select candidate literals from rules_df ----
		if rules_df is None or len(rules_df) == 0:
			return {"found_solution": False, "success": False,
					"reason": "rules_df is empty; run get_rules(...) first and pass its output here."}

		rdf = rules_df
		needed = {"component_type", "rule_index", "is_negated", "rule_expression"}
		if not needed.issubset(set(rdf.columns)):
			return {"found_solution": False, "success": False,
					"reason": "rules_df must include component_type, rule_index, is_negated, rule_expression."}

		rdf_rules = rdf[rdf["component_type"] == "rule"].copy()
		rdf_rules = rdf_rules.dropna(subset=["rule_index", "is_negated"])
		if rdf_rules.empty:
			fire_empty = np.zeros(n, dtype=bool)
			metrics = self._signed_rule_metrics_against_target(fire_empty, yb, True, eps=1e-12)
			yhat_combo = fire_empty.astype(np.int8)
			out = {
				"found_solution": True,
				"success": bool(np.array_equal(yhat_combo, yb)),
				"perfect_match": bool(np.array_equal(yhat_combo, yb)),
				"target": target_name,
				"problem": "classification" if is_clf else "regression",
				"selected_rule_indices": [],
				"selected_literals": [],
				"expression": "FALSE",
				"dataset_coverage": float(fire_empty.mean()),
				"P(target=1|fire)": metrics[0],
				"R(fire|target=1)": metrics[1],
				"F1(target=1|fire)": metrics[2],
				"Lift(target=1|fire)": metrics[3],
				"Acc": metrics[4],
				"TPR": metrics[5],
				"TNR": metrics[6],
				"BalancedAcc": metrics[7],
				"MCC": metrics[8],
				"chosen_from": "empty",
			}
			if not is_clf:
				out.update({
					"reg_tail_q": float(tail_q),
					"reg_event_threshold": float(evt_threshold),
					"reg_event_direction": evt_direction,
				})
				if y_cont is not None:
					seg = self._segmentation_stats(fire_empty, y_cont, sign_is_positive, tail_q=tail_q, eps=1e-12)
					out.update({
						"r_pb": seg[0],
						"cohen_d": seg[1],
						"mean_shift": seg[2],
						"r2_seg": seg[3],
						"prec_tail": seg[4],
						"rec_tail": seg[5],
						"f1_tail": seg[6],
						"lift_tail": seg[7],
					})
			if return_pred:
				out["y_target"] = yb
				out["yhat_combo"] = yhat_combo
			return out

		if max_selected is not None and max_selected <= 0:
			max_selected = 0

		rule_idx_all = rdf_rules["rule_index"].astype(int).to_numpy()
		is_neg_all = rdf_rules["is_negated"].astype(bool).to_numpy()
		literal_expr_all = rdf_rules["rule_expression"].astype(str).to_list()
		m_all = rule_idx_all.size


		# ---- Optional: importance metrics if present in rules_df ----
		# These are rule-level scores (not target-specific). They are only used for
		# seeding / screening when requested via greedy_seed_metrics.
		importance_all = None
		weighted_importance_all = None
		if hasattr(rdf_rules, "columns"):
			if "importance" in rdf_rules.columns:
				try:
					importance_all = pd.to_numeric(rdf_rules["importance"], errors="coerce").to_numpy(dtype=np.float64)
				except Exception:
					importance_all = np.asarray(rdf_rules["importance"], dtype=np.float64)
				if importance_all.shape[0] != m_all:
					importance_all = None

			# weighted importance has a few possible column names historically
			w_cols = [
				"importance_weighted_by_gain",
				"weighted_importance",
				"weightedimportance",
			]
			for c in w_cols:
				if c in rdf_rules.columns:
					try:
						weighted_importance_all = pd.to_numeric(rdf_rules[c], errors="coerce").to_numpy(dtype=np.float64)
					except Exception:
						weighted_importance_all = np.asarray(rdf_rules[c], dtype=np.float64)
					break
			if weighted_importance_all is not None and weighted_importance_all.shape[0] != m_all:
				weighted_importance_all = None
		# ---- MCC helper (vectorized for screening) ----
		def _mcc_from_counts(tp, fp, tn, fn, eps=1e-12):
			tp = tp.astype(np.float64, copy=False)
			fp = fp.astype(np.float64, copy=False)
			tn = tn.astype(np.float64, copy=False)
			fn = fn.astype(np.float64, copy=False)
			num = tp * tn - fp * fn
			den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
			return num / (np.sqrt(den) + eps)

		# ==========================
		# 1) SCREEN candidates by single-literal MCC (vectorized, no L matrix)
		# ==========================
		Z_pos = Z[y_pos][:, rule_idx_all]
		Z_neg = Z[~y_pos][:, rule_idx_all]

		pos_ones = Z_pos.sum(axis=0).astype(np.int64)
		neg_ones = Z_neg.sum(axis=0).astype(np.int64)

		tp_single = pos_ones.copy()
		fp_single = neg_ones.copy()

		if is_neg_all.any():
			neg_mask = is_neg_all
			tp_single[neg_mask] = P - pos_ones[neg_mask]
			fp_single[neg_mask] = N - neg_ones[neg_mask]

		tn_single = N - fp_single
		fn_single = P - tp_single
		mcc_single = _mcc_from_counts(tp_single, fp_single, tn_single, fn_single)

		# ---- Single-literal metrics (from counts; used for screening + optional multi-seed greedy) ----
		eps = 1e-12
		with np.errstate(divide="ignore", invalid="ignore"):
			cov_single = (tp_single + fp_single) / float(n)
			pre_single = tp_single / (tp_single + fp_single + eps)   # P(target=1|fire)
			rec_single = tp_single / (float(P) + eps)               # R(fire|target=1)
			f1_single = 2.0 * pre_single * rec_single / (pre_single + rec_single + eps)
			base_pos = float(P) / float(n)
			lift_single = pre_single / (base_pos + eps)
			acc_single = (tp_single + tn_single) / float(n)
			tpr_single = rec_single
			tnr_single = tn_single / (float(N) + eps)
			balacc_single = 0.5 * (tpr_single + tnr_single)
			fpr_single = fp_single / (float(N) + eps)
			fnr_single = fn_single / (float(P) + eps)

		metric_single = {
			"MCC": mcc_single,
			"F1": f1_single,
			"Lift": lift_single,
			"BalancedAcc": balacc_single,
			"Acc": acc_single,
			"Precision": pre_single,
			"Recall": rec_single,
			"Coverage": cov_single,
			"TPR": tpr_single,
			"TNR": tnr_single,
			"FPR": fpr_single,
			"FNR": fnr_single,
		}

		# ---- Optional: add importance-based seed metrics (only if present) ----
		if importance_all is not None and np.isfinite(importance_all).any():
			metric_single["Importance"] = importance_all
		if weighted_importance_all is not None and np.isfinite(weighted_importance_all).any():
			metric_single["WeightedImportance"] = weighted_importance_all

		def _norm_metric_name(name):
			if name is None:
				return ""
			s = str(name).strip().lower()
			s = s.replace(" ", "").replace("_", "")
			if s in ("mcc",):
				return "MCC"
			if s in ("f1", "f1target=1|fire", "f1target1|fire"):
				return "F1"
			if s in ("lift", "lifttarget=1|fire", "lifttarget1|fire"):
				return "Lift"
			if s in ("balancedacc", "balancedaccuracy", "balacc"):
				return "BalancedAcc"
			if s in ("acc", "accuracy"):
				return "Acc"
			if s in ("precision", "p", "pre", "ptarget=1|fire", "ptarget1|fire"):
				return "Precision"
			if s in ("recall", "r", "rec", "rfire|target=1", "rfire|target1", "recallfire|target=1"):
				return "Recall"
			if s in ("coverage", "datasetcoverage", "fire", "fire_rate", "firingrate"):
				return "Coverage"
			if s in ("tpr", "sensitivity", "recallpos"):
				return "TPR"
			if s in ("tnr", "specificity"):
				return "TNR"
			if s in ("fpr", "falsepositive_rate", "falsepositiverate"):
				return "FPR"
			if s in ("fnr", "falsenegative_rate", "falsenegativerate"):
				return "FNR"
			if s in ("importance", "imp", "abscoef", "abscoefficient"):
				return "Importance"
			if s in ("weightedimportance", "importanceweightedbygain", "importanceweighted", "impweighted", "weightedimp", "gainweightedimportance", "importancegainweighted"):
				return "WeightedImportance"
			return str(name)

		def _direction_is_asc(metric_name):
			if greedy_seed_metric_directions is None:
				return metric_name in ("FPR", "FNR")
			# try direct + normalized keys
			d = greedy_seed_metric_directions.get(metric_name, None)
			if d is None:
				d = greedy_seed_metric_directions.get(str(metric_name).lower(), None)
			if d is None:
				d = greedy_seed_metric_directions.get(str(metric_name).replace(" ", ""), None)
			if d is None:
				return False
			ds = str(d).strip().lower()
			return ds in ("asc", "ascending", "min", "minimize", "lower", "smallest")

		# ---- Screen by single-literal MCC (existing behavior) ----
		order = np.argsort(mcc_single)[::-1]
		if np.isfinite(min_single_mcc):
			ok = np.isfinite(mcc_single[order]) & (mcc_single[order] >= min_single_mcc)
			order = order[ok]
		if topk_screen is not None and topk_screen > 0 and order.size > topk_screen:
			order = order[:topk_screen]

		if order.size == 0 and m_all > 0:
			order = np.array([int(np.nanargmax(mcc_single))], dtype=int)

		# ---- Optional: union in top literals by other metrics (Variant A) ----
		greedy_seed_metrics_norm = ["MCC"]
		if greedy_seed_metrics is not None:
			if isinstance(greedy_seed_metrics, (str, bytes)):
				glist = [greedy_seed_metrics]
			else:
				glist = list(greedy_seed_metrics)
			greedy_seed_metrics_norm = [_norm_metric_name(x) for x in glist]
			greedy_seed_metrics_norm = [x for x in greedy_seed_metrics_norm if x in metric_single]
			if "MCC" not in greedy_seed_metrics_norm:
				greedy_seed_metrics_norm = ["MCC"] + greedy_seed_metrics_norm

			extra = []
			pool_k = int(greedy_seed_pool_k) if (greedy_seed_pool_k is not None and int(greedy_seed_pool_k) > 0) else 0
			k_seed = int(greedy_seed_topk) if (greedy_seed_topk is not None and int(greedy_seed_topk) > 0) else 1

			for met in greedy_seed_metrics_norm:
				vals = metric_single[met]
				finite = np.isfinite(vals)
				if not finite.any():
					continue
				asc = _direction_is_asc(met)
				if asc:
					vv = np.where(finite, vals, np.inf)
					idxs = np.argsort(vv)
				else:
					vv = np.where(finite, vals, -np.inf)
					idxs = np.argsort(vv)[::-1]

				extra.extend([int(i) for i in idxs[:k_seed]])
				if pool_k > k_seed:
					extra.extend([int(i) for i in idxs[:pool_k]])

			if extra:
				order_list = list(order.astype(int))
				have = set(order_list)
				for i in extra:
					if i not in have:
						order_list.append(i)
						have.add(i)
				order = np.asarray(order_list, dtype=int)
		rule_idx = rule_idx_all[order]
		is_neg = is_neg_all[order]
		literal_expr = [literal_expr_all[i] for i in order]
		m = rule_idx.size

		j_best = int(np.nanargmax(mcc_single[order])) if m > 0 else None
		best_single_mcc = float(mcc_single[order][j_best]) if j_best is not None else -np.inf

		# ==========================
		# 2) Bitset machinery (Python int)
		# ==========================
		def _mask_to_int(mask_bool: np.ndarray) -> int:
			packed = np.packbits(mask_bool.astype(np.uint8, copy=False), bitorder="little")
			return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)

		all_bits = (1 << n) - 1 if n > 0 else 0

		pos_bits = _mask_to_int(y_pos) & all_bits
		neg_bits = (all_bits ^ pos_bits) & all_bits

		lit_bits = []
		for j, neg in zip(rule_idx, is_neg):
			if not neg:
				mask = (Z[:, j] == 1)
			else:
				mask = (Z[:, j] == 0)
			lit_bits.append(_mask_to_int(mask) & all_bits)

		# ==========================
		# 3) Baselines (empty + single)
		# ==========================
		fire_empty_bits = 0
		fire_empty = np.zeros(n, dtype=bool)
		m_empty_full = self._signed_rule_metrics_against_target(fire_empty, yb, True, eps=1e-12)
		mcc_empty = float(m_empty_full[8]) if np.isfinite(m_empty_full[8]) else -np.inf

		best_fire_bits = fire_empty_bits
		best_sel_cols = []
		best_metrics = m_empty_full
		best_mcc = mcc_empty
		best_origin = "empty"
		greedy_best_mcc = None

		if j_best is not None and best_single_mcc > best_mcc:
			best_sel_cols = [j_best]
			best_fire_bits = lit_bits[j_best]
			best_mcc = best_single_mcc
			best_origin = "single"
			best_metrics = None

		# ==========================
		# 4) Greedy OR (bitset, very fast inner loop)
		# ==========================
		def _mcc_scalar(tp, fp, tn, fn):
			# scalar MCC (avoid giant int products -> compute den in float space)
			a = tp + fp
			b = tp + fn
			c = tn + fp
			d = tn + fn
			den = float(a) * float(b) * float(c) * float(d)
			if den <= 0.0 or not np.isfinite(den):
				return -np.inf
			num = float(tp * tn - fp * fn)
			return num / (math.sqrt(den) + 1e-12)

		def _run_greedy(seed_col=None, desc=None):
			# seed_col is an index in the screened candidate list [0..m-1]
			if seed_col is None:
				current_fire = fire_empty_bits
				current_tp = 0
				current_fp = 0
				current_mcc = float(mcc_empty)
				available = np.ones(m, dtype=bool)
				chosen_cols = []
			else:
				current_fire = lit_bits[int(seed_col)]
				current_tp = (current_fire & pos_bits).bit_count()
				current_fp = (current_fire & neg_bits).bit_count()
				tn0 = N - current_fp
				fn0 = P - current_tp
				current_mcc = float(_mcc_scalar(current_tp, current_fp, tn0, fn0))
				available = np.ones(m, dtype=bool)
				available[int(seed_col)] = False
				chosen_cols = [int(seed_col)]

			it = range(m)
			if show_progress:
				it = tqdm(it, desc=(desc or "Greedy OR (bitset MCC)"), leave=False)

			for _ in it:
				if max_selected is not None and len(chosen_cols) >= max_selected:
					break

				not_fire = (~current_fire) & all_bits
				if not_fire == 0:
					break

				best_k = -1
				best_k_mcc = -np.inf

				for k in range(m):
					if not available[k]:
						continue
					added = lit_bits[k] & not_fire
					if added == 0:
						continue

					tp_add = (added & pos_bits).bit_count()
					fp_add = (added & neg_bits).bit_count()

					tp_new = current_tp + tp_add
					fp_new = current_fp + fp_add
					fn_new = P - tp_new
					tn_new = N - fp_new

					mcc_new = _mcc_scalar(tp_new, fp_new, tn_new, fn_new)

					if mcc_new > best_k_mcc:
						best_k_mcc = mcc_new
						best_k = k

				if best_k < 0 or (not np.isfinite(best_k_mcc)) or (best_k_mcc - current_mcc) <= 0.0:
					break

				added = lit_bits[best_k] & ((~current_fire) & all_bits)
				current_tp += (added & pos_bits).bit_count()
				current_fp += (added & neg_bits).bit_count()
				current_fire |= lit_bits[best_k]
				current_mcc = float(best_k_mcc)

				available[best_k] = False
				chosen_cols.append(best_k)

			return chosen_cols, current_fire, float(current_mcc)

		# ---- Multi-seed greedy (Variant A) ----
		greedy_best_seed_metric = None
		greedy_best_seed_literal = None
		greedy_mcc_by_seed_metric = None
		greedy_runs_detail = None

		if (max_selected is None or max_selected > 1) and m > 0:
			seeds_to_try = []
			# Always include the original MCC-greedy path (start empty).
			seeds_to_try.append(("MCC", None))

			if greedy_seed_metrics is not None:
				k_seed = int(greedy_seed_topk) if (greedy_seed_topk is not None and int(greedy_seed_topk) > 0) else 1
				for met in greedy_seed_metrics_norm:
					if met == "MCC":
						continue
					vals = metric_single[met][order]
					finite = np.isfinite(vals)
					if not finite.any():
						continue
					asc = _direction_is_asc(met)
					if asc:
						vv = np.where(finite, vals, np.inf)
						idxs = np.argsort(vv)
					else:
						vv = np.where(finite, vals, -np.inf)
						idxs = np.argsort(vv)[::-1]
					for j in idxs[:k_seed]:
						seeds_to_try.append((met, int(j)))

			# de-dup seeds
			seen = set()
			seeds_unique = []
			for met, seed_col in seeds_to_try:
				key = (met, int(seed_col) if seed_col is not None else -1)
				if key in seen:
					continue
				seen.add(key)
				seeds_unique.append((met, seed_col))

			greedy_mcc_by_seed_metric = {}
			if return_greedy_runs:
				greedy_runs_detail = []

			best_g_cols = None
			best_g_fire = None
			best_g_mcc = -np.inf
			best_g_seed_metric = None
			best_g_seed_literal = None

			for met, seed_col in seeds_unique:
				desc = f"Greedy OR seed={met}" if (show_progress and greedy_seed_metrics is not None) else "Greedy OR (bitset MCC)"
				cols, fire_bits, mcc_val = _run_greedy(seed_col=seed_col, desc=desc)

				prev = greedy_mcc_by_seed_metric.get(met, -np.inf)
				if mcc_val > prev:
					greedy_mcc_by_seed_metric[met] = float(mcc_val)

				if return_greedy_runs:
					seed_lit = None
					seed_rule = None
					if seed_col is not None:
						seed_lit = literal_expr[int(seed_col)]
						seed_rule = int(rule_idx[int(seed_col)])
					greedy_runs_detail.append({
						"seed_metric": met,
						"seed_rule_index": seed_rule,
						"seed_literal": seed_lit,
						"n_literals": int(len(cols)),
						"MCC": float(mcc_val),
					})

				if (best_g_cols is None) or (mcc_val > best_g_mcc) or (mcc_val == best_g_mcc and len(cols) < len(best_g_cols)):
					best_g_cols = cols
					best_g_fire = fire_bits
					best_g_mcc = float(mcc_val)
					best_g_seed_metric = met
					best_g_seed_literal = literal_expr[int(seed_col)] if seed_col is not None else None

			# record best greedy result even if it doesn't beat single/empty
			greedy_best_mcc = float(best_g_mcc) if best_g_cols is not None else float("nan")
			greedy_best_seed_metric = best_g_seed_metric
			greedy_best_seed_literal = best_g_seed_literal

			if best_g_cols is not None:
				if best_g_mcc > best_mcc or (best_g_mcc == best_mcc and len(best_g_cols) < len(best_sel_cols)):
					best_sel_cols = best_g_cols
					best_fire_bits = best_g_fire
					best_mcc = float(best_g_mcc)
					best_origin = "greedy"
					best_metrics = None
		# ==========================
		# 5) Final full metrics + output
		# ==========================
		if n > 0:
			nbytes = (n + 7) // 8
			b = int(best_fire_bits).to_bytes(nbytes, byteorder="little", signed=False)
			fire_arr = np.frombuffer(b, dtype=np.uint8)
			fire_bool = np.unpackbits(fire_arr, bitorder="little")[:n].astype(bool, copy=False)
		else:
			fire_bool = np.zeros(0, dtype=bool)

		if best_metrics is None:
			best_metrics = self._signed_rule_metrics_against_target(fire_bool, yb, True, eps=1e-12)

		yhat_combo = fire_bool.astype(np.int8)

		selected_literals = [literal_expr[c] for c in best_sel_cols]
		expr = " OR ".join(f"({s})" for s in selected_literals) if selected_literals else "FALSE"

		out = {
			"found_solution": True,
			"success": bool(np.array_equal(yhat_combo, yb)),
			"perfect_match": bool(np.array_equal(yhat_combo, yb)),
			"target": target_name,
			"problem": "classification" if is_clf else "regression",
			"selected_rule_indices": [int(rule_idx[c]) for c in best_sel_cols],
			"selected_literals": selected_literals,
			"expression": expr,
			"dataset_coverage": float(fire_bool.mean()) if n > 0 else 0.0,
			"P(target=1|fire)": best_metrics[0],
			"R(fire|target=1)": best_metrics[1],
			"F1(target=1|fire)": best_metrics[2],
			"Lift(target=1|fire)": best_metrics[3],
			"Acc": best_metrics[4],
			"TPR": best_metrics[5],
			"TNR": best_metrics[6],
			"BalancedAcc": best_metrics[7],
			"MCC": best_metrics[8],
			"chosen_from": best_origin,
			"empty_MCC": float(mcc_empty),
			"best_single_MCC": float(best_single_mcc),
			"best_greedy_MCC": float(greedy_best_mcc) if greedy_best_mcc is not None else float("nan"),
			"screen_topk": int(m),
		}

		if greedy_seed_metrics is not None:
			out.update({
				"greedy_seed_metrics": list(greedy_seed_metrics_norm),
				"greedy_winning_seed_metric": greedy_best_seed_metric,
				"greedy_winning_seed_literal": greedy_best_seed_literal,
				"best_greedy_MCC_by_seed_metric": {k: float(v) for k, v in (greedy_mcc_by_seed_metric or {}).items()},
			})
			if return_greedy_runs and greedy_runs_detail is not None:
				out["greedy_runs"] = greedy_runs_detail

		if not is_clf:
			out.update({
				"reg_tail_q": float(tail_q),
				"reg_event_threshold": float(evt_threshold),
				"reg_event_direction": evt_direction,
			})
			# segmentation stats vs continuous target
			if y_cont is not None:
				seg = self._segmentation_stats(fire_bool, y_cont, sign_is_positive, tail_q=tail_q, eps=1e-12)
				out.update({
					"r_pb": seg[0],
					"cohen_d": seg[1],
					"mean_shift": seg[2],
					"r2_seg": seg[3],
					"prec_tail": seg[4],
					"rec_tail": seg[5],
					"f1_tail": seg[6],
					"lift_tail": seg[7],
				})

		if return_pred:
			out["y_target"] = yb
			out["yhat_combo"] = yhat_combo
			if not is_clf:
				out["y_target_cont"] = y_cont

		return out



	@staticmethod
	def _signed_rule_metrics_against_target(fire, y_target_bin, sign_is_positive, eps=1e-12):
		"""
		fire: bool mask where component "fires"
		y_target_bin: {0,1} target labels (here: ŷ_model binarized)
		sign_is_positive: coef > 0
			- if True: coefficient_sign class is 1 when fire
			- if False: coefficient_sign class is 0 when fire
		Returns: list of metrics (all vs y_target_bin)
		"""
		fire = np.asarray(fire, dtype=bool)
		yb = np.asarray(y_target_bin, dtype=np.int8).reshape(-1)
		n = yb.size
		if n == 0:
			return [np.nan] * 9

		n_fire = int(fire.sum())
		sign_fire = 1 if sign_is_positive else 0

		# Metrics conditional on fire (precision-like)
		if n_fire > 0:
			pre_signed = float((yb[fire] == sign_fire).mean())
		else:
			pre_signed = np.nan

		# Recall-like for the coefficient_sign class (coverage of that class by firing)
		denom_signed = int((yb == sign_fire).sum())
		rec_signed = (float(((yb == sign_fire) & fire).sum()) / denom_signed) if denom_signed > 0 else np.nan

		# F1 & lift (still meaningful vs ŷ_model)
		f1_signed = (
			(2.0 * pre_signed * rec_signed / (pre_signed + rec_signed + eps))
			if (np.isfinite(pre_signed) and np.isfinite(rec_signed) and (pre_signed + rec_signed) > 0)
			else np.nan
		)
		base_signed = float((yb == sign_fire).mean())
		lift_signed = (pre_signed / (base_signed + eps)) if np.isfinite(pre_signed) else np.nan

		# Agreement of full signed rule-prediction vs target: ŷ_rule = fire if coef>0 else ~fire
		y_rule = fire if sign_is_positive else ~fire
		y_rule = y_rule.astype(bool)

		y1 = (yb == 1)
		tp = float((y_rule & y1).sum())
		fp = float((y_rule & ~y1).sum())
		fn = float((~y_rule & y1).sum())
		tn = float((~y_rule & ~y1).sum())

		acc = (tp + tn) / n
		tpr = (tp / (tp + fn + eps)) if (tp + fn) > 0 else np.nan   # coverage of ŷ_model=1 correctly matched
		tnr = (tn / (tn + fp + eps)) if (tn + fp) > 0 else np.nan   # coverage of ŷ_model=0 correctly matched
		bal_acc = 0.5 * (tpr + tnr) if (np.isfinite(tpr) and np.isfinite(tnr)) else np.nan

		denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
		mcc = 0.0 if denom <= 0 else (tp * tn - fp * fn) / denom

		return [
			pre_signed, rec_signed, f1_signed, lift_signed,
			acc, tpr, tnr, bal_acc, mcc
		]

	@staticmethod
	def _segmentation_stats(fire, target, sign_is_positive, tail_q=0.90, eps=1e-12):
		"""
		Regression-style stats computed vs target (here: ŷ_model continuous).
		Returns: [r_pb, d_cohen, mean_shift, r2_seg, prec_tail, rec_tail, f1_tail, lift_tail]
		"""
		fire = np.asarray(fire, dtype=bool)
		t = np.asarray(target, dtype=np.float64).reshape(-1)
		n = t.size
		if n == 0:
			return [np.nan] * 8

		n1 = int(fire.sum())
		n0 = n - n1
		p = (n1 / n) if n > 0 else np.nan

		if n1 > 0:
			mu1 = float(np.nanmean(t[fire]))
			mse1 = float(np.nanmean((t[fire] - mu1) ** 2))
			var1 = float(np.nanvar(t[fire], ddof=1)) if n1 > 1 else np.nan
		else:
			mu1 = mse1 = var1 = np.nan

		if n0 > 0:
			mu0 = float(np.nanmean(t[~fire]))
			mse0 = float(np.nanmean((t[~fire] - mu0) ** 2))
			var0 = float(np.nanvar(t[~fire], ddof=1)) if n0 > 1 else np.nan
		else:
			mu0 = mse0 = var0 = np.nan

		mean_shift = (mu1 - mu0) if (np.isfinite(mu1) and np.isfinite(mu0)) else np.nan

		t_std = float(np.nanstd(t, ddof=1)) if n > 1 else 0.0
		r_pb = ((mu1 - mu0) * np.sqrt(p * (1 - p)) / (t_std + eps)) if (t_std > 0 and np.isfinite(p)) else np.nan

		if n1 > 1 and n0 > 1 and np.isfinite(var1) and np.isfinite(var0):
			s_pooled = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
			d_cohen = (mu1 - mu0) / (s_pooled + eps)
		else:
			d_cohen = np.nan

		if np.isfinite(mse1) and np.isfinite(mse0) and np.isfinite(p):
			var_t = float(np.nanvar(t)) + eps
			mse_seg = p * mse1 + (1.0 - p) * mse0
			r2_seg = 1.0 - (mse_seg / var_t)
		else:
			r2_seg = np.nan

		# Signed tail event on target (ŷ_model)
		if n > 0:
			t_hi = float(np.nanquantile(t, tail_q))
			t_lo = float(np.nanquantile(t, 1.0 - tail_q))
			evt = (t >= t_hi) if sign_is_positive else (t <= t_lo)
			evt_total = int(evt.sum())
			base_evt = evt_total / n
		else:
			evt = None
			evt_total = 0
			base_evt = np.nan

		if n1 > 0 and evt is not None and evt_total > 0:
			tp = int((fire & evt).sum())
			prec = tp / n1
			rec = tp / evt_total
			f1 = (2.0 * prec * rec / (prec + rec + eps)) if (prec + rec) > 0 else np.nan
			lift = (prec / (base_evt + eps)) if np.isfinite(base_evt) else np.nan
		else:
			prec = rec = f1 = lift = np.nan

		return [r_pb, d_cohen, mean_shift, r2_seg, prec, rec, f1, lift]


###############
### Test cases
###############
assert Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
])) > Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])), "Failed test case 1"
###############
assert not Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.6, '<='),
	RuleCondition(2, 0.7, '>'),
])) > Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])), "Failed test case 2"
###############
assert Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
	RuleCondition(3, 0.7, '>'),
])) < Rule(List([
	RuleCondition(0, 0.5, '>'),
	RuleCondition(1, 0.9, '<='),
	RuleCondition(2, 0.7, '>'),
])), "Failed test case 3"