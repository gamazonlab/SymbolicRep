"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy, deepcopy

import numpy as np
from sklearn.utils.random import sample_without_replacement

#### ode change
#from jax.experimental.ode import odeint
#from jax import grad
#import jax
#from sympy import sin, cos, sympify
#from torchdiffeq import odeint
#import torch
from scipy.integrate import odeint
from .functions import FOG, FOXX, FOGH, FOGC, FOCG, FOGX, FOXG, FOXC, FOCX, ODEF
####

from .functions import _Function, _function_map
from .utils import check_random_state

from scipy.optimize import curve_fit

class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None,
                 p0=None,
                 prior=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.p0 = p0
        self.prior = prior
        self.fit_index = []
        self.p_ij = {}
        self.H_j = 0.0
        self.H_Rj = 0.0
        self.max_exp = 6
        self.min_exp = -3
        self.min_exp = -6
        self.truncate = []
        
        self.sum = _function_map['sum']
        self.prod = _function_map['prod']
        self.pow = _function_map['pow']
        self.mul = _function_map['mul']
        self.affine = _function_map['affine']
        
        if self.prior is not None: 
            self.prior_functions = []
            self.CDF = np.zeros(len(self.prior))
            val = 0.0 
            i = 0
            for f in self.prior:
                if isinstance(f, _Function):
                    self.prior_functions.append(f)
                elif isinstance(f, str):
                    self.prior_functions.append(_function_map[f])
                else:
                    self.prior_functions.append(f)
		        
                self.CDF[i] = self.prior[f]
                i += 1
            self.CDF = np.cumsum(self.CDF / self.CDF.sum())
        
        #print(self.CDF)
        #print(self.included_f_ids)    
        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        elif self.prior is None:
            # Create a naive random program
            self.program, self.p0 = self.build_program(random_state)
        else :
            # Create a random program with the prior distribution
            #print(prior)
            self.program, self.p0 = self.build_program_with_prior(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None            

    def print_program(self, program=None, p0=None, pflag=True):
    
        if program is None:
            program = self.program
        
        if p0 is None:
            p0 = self.p0
            
        stack = []
        if pflag:
            for (p, n) in zip(p0, program):
                if isinstance(n, _Function):
                    if p != 1.0:
                        stack.append(p)
                    stack.append(n.name)
                elif isinstance(n, int):
                    if p != 1.0:
                        stack.append(p)
                    stack.append(n)
                else:
                    stack.append(n)
        else:
            for n in program:
                if isinstance(n, _Function):
                    stack.append(n.name)
                elif isinstance(n, int):
                    stack.append(n)
                else:
                    stack.append(n)        
        
        print(stack)
        print()
        print(self.__latex_str__())
        
    def __latex_str__(self, program=None, p0=None, pflag=True):
        if program is None:
            program = self.program
        if p0 is None:
            p0 = self.p0

        stack = []
        for (p, n) in zip(p0, program): 
            p = float("{:.3f}".format(p))
            if p == 1.0:
                p = ''
            elif p == -1.0:
                p = '-'
            else:
                p = str(p)
            if isinstance(n, list):
                s = stack.pop()
                if n[0] == 1:
                    if s[1] == 'affine' or s[1] == 'lin':
                        if s[0] != '':
                            pr = s[0] + '(' + s[2] + ')'
                        else:
                            pr = s[2] 
                    elif s[1] == 'sum':
                        pr = ''
                        for ss in s[2:]:
                            pr += ss + '+' 
                        pr = pr[:-1]
                        if s[0] != '':
                            pr = s[0] + '(' + pr + ')'
                    elif s[1] == 'prod':
                        pr = ''
                        for ss in s[2:]:
                            pr += ss + ' ' 
                        pr = pr[:-1]
                        if s[0] != '':
                            pr = s[0] + '(' + pr + ')'
                    elif s[1] == 'sqrt':
                        pr = s[0] + '\\sqrt{' + s[2] + '}'
                    else:
                        pr = s[0] + '\\' + s[1] + '{(' + s[2] + ')}'
                elif n[0] == -1:
                    if s[1] == 'affine' or s[1] == 'lin':
                        pr = s[2]
                    elif s[1] == 'sum':
                        pr = ''
                        for ss in s[2:]:
                            pr += ss + '+' 
                        pr = pr[:-1]
                    elif s[1] == 'prod':
                        pr = ''
                        for ss in s[2:]:
                            pr += ss + ' ' 
                        pr = pr[:-1]
                    elif s[1] == 'sqrt':
                        pr = '\\sqrt{' + s[2] + '}'
                    else:
                        pr = '\\' + s[1] + '{(' + s[2] + ')}'
                    if s[0] == '':
                        pr = '\\frac{1}{' + pr + '}'
                    elif s[0] == '-':
                        pr = '-\\frac{1}{' + pr + '}'
                    else:
                        pr = '\\frac{' + s[0] + '}{' + pr + '}'
                elif n[0] == 0:
                    if s[0] == '':
                        pr = '1.0'
                    elif s[0] == '-':
                        pr = '-1.0'
                    else:
                        pr = s[0]
                elif n[0] > 0:
                    if s[1] == 'affine' or s[1] == 'lin':
                        pr = s[0] + '(' + s[2] + ')^' + str(n[0])
                    elif s[1] == 'sum':
                        pr = s[0] + '('
                        for ss in s[2:]:
                            pr += ss + '+' 
                        pr = pr[:-1] + ')^' + str(n[0])
                    elif s[1] == 'prod':
                        pr = s[0] +  '('
                        for ss in s[2:]:
                            pr += ss + ' ' 
                        pr = pr[:-1] + ')^' + str(n[0])
                    elif s[1] == 'sqrt':
                        pr = s[0] + '(\\sqrt{' + s[2] + '})^' + str(n[0])
                    else:
                        pr = s[0] + '\\' + s[1] + '^' + str(n[0]) + '{(' + s[2] + ')}'
                elif n[0] < 0:
                    if s[1] == 'affine' or s[1] == 'lin':
                        pr = '(' + s[2] + ')^' + str(abs(n[0]))
                    elif s[1] == 'sum':
                        pr = '('
                        for ss in s[2:]:
                            pr += ss + '+' 
                        pr = pr[:-1] + ')^' + str(abs(n[0]))
                    elif s[1] == 'prod':
                        pr = '('
                        for ss in s[2:]:
                            pr += ss + ' ' 
                        pr = pr[:-1] + ')^' + str(abs(n[0]))
                    elif s[1] == 'sqrt':
                        pr = '(\\sqrt{' + s[2] + '})^' + str(abs(n[0]))
                    else:
                        pr = '\\' + s[1] + '^' + str(abs(n[0])) + '{(' + s[2] + ')}'
                    if s[0] == '':
                        pr = '\\frac{1}{' + pr + '}'
                    elif s[0] == '-':
                        pr = '-\\frac{1}{' + pr + '}'
                    else:
                        pr = '\\frac{s[0]}{' + pr + '}'
                if len(stack) == 0:
                    return pr
                else:
                    stack[-1].append(pr)
            elif isinstance(n, _Function):
                stack.append([p, n.name])
            elif isinstance(n, int):
                stack[-1].append(p+'x_'+str(n))
            elif isinstance(n, float):
                stack[-1].append("{:.2f}".format(n))
        return None        

    def snp_location(self, snp, program=None, times=False):
        
        if program == None:
            program = self.program    
        
        flag = False
        
        balance = 0
        if isinstance(snp[0], int):
            index = 0
            for i in range(len(program)-len(snp)+1):
                if program[i:i+len(snp)] == snp and balance == 0:
                    index = i 
                    flag = True
                    break
                if isinstance(program[i], list):
                    balance -= 1
                elif isinstance(program[i], int):
                    if program[i] > snp[0] and balance == 0:
                        index = i
                elif isinstance(program[i], _Function):
                    balance += 1
        elif isinstance(snp[0], float):
            index = 0
        else:
            index = len(program)
            if times:
                irange = len(program) - len(snp)
            else:
                irange = len(program) - len(snp) + 1
                
            for i in range(irange):
                if times:
                    if isinstance(program[i+len(snp)], list) and program[i:i+len(snp)] == snp and balance == 0:
                        index = i
                        flag = True
                        break
                elif program[i:i+len(snp)] == snp and balance == 0:
                    index = i 
                    flag = True
                    break
                if isinstance(program[i], _Function):
                    if program[i].name > snp[0].name and balance == 0:
                        index = i
                    balance += 1
                elif isinstance(program[i], list):
                    balance -= 1
              
        
        #print("snp")
        #print(snp_str)
        #print(program_str)
        #print(index)
        #print(flag)
        #print()             
        return flag, index
                
    def build_program(self, random_state, init_depth=0):
        
        sumprod = random_state.uniform()
        if sumprod < 0.5:
            program, depth = self.program_sum(random_state, init_depth)
        else:
            program, depth = self.program_times(random_state, init_depth)
            
        p0 = self.p0_init(program, random_state)
        
        #self.print_program(program, pflag=False)
        
        return program, p0
        

    def program_sum(self, random_state, init_depth=0):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        
        function_list = list(range(len(self.function_set)))
        
        feature_list = list(range(self.n_features))

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(function_list))
        function = function_list[function]
        #function_list.remove(function)
        function = self.function_set[function]
        var = random_state.randint(self.n_features)
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp)
            if p == 0:
                p = 1
        program = [function, var, [p]]
        depth = max(2*abs(p), init_depth)
        #program = [function, var, [1]]
        #depth = max(3, init_depth)
        
        single_fun = True
        
        

        while depth < max_depth:
            n_choice = len(feature_list) + len(function_list) + 2
            #n_choice = len(function_list) + 2
            choice = random_state.randint(n_choice)
            #print(n_choice, choice, len(function_list))
            # Determine if we are adding a function or terminal
            # if method == 'full' or choice < len(function_list):
            if choice < len(function_list):
                function = random_state.randint(len(function_list))
                function = function_list[function]
                #function_list.remove(function)
                function = self.function_set[function]
                var = random_state.randint(self.n_features)
                p = 1
                if random_state.uniform() < 0.025:
                    p = random_state.randint(self.min_exp, self.max_exp)
                    if p == 0:
            	         p = 1
                flag, index = self.snp_location([function, var, [p]], program)
                #flag, index = self.snp_location([function, var, [1]], program)
                
                if not flag:
                    program = program[:index] + [function, var, [p]] + program[index:]
                    depth += 2*abs(p)
                    #program = program[:index] + [function, var, [1]] + program[index:]
                    #depth += 3
                    single_fun = False
                
            elif choice == n_choice - 2:
                internal_program, internal_depth = self.program_times(random_state, depth)
                flag, index = self.snp_location(internal_program, program)
                if not flag:
                    program = program[:index] + internal_program + program[index:]                    
                    depth += internal_depth
                    single_fun = False
                    
            elif choice == n_choice - 1:
                function = random_state.randint(len(self.function_set))
                function = function_list[function]
                function = self.function_set[function]
                internal_program, internal_depth = self.program_sum(random_state, depth)
                
                #print("build", internal_program)
                p = 1
                if random_state.uniform() < 0.025:
                    p = random_state.randint(self.min_exp, self.max_exp)
                    if p == 0:
            	         p = 1
                flag, index = self.snp_location([function, internal_program, [p]], program)
                #flag, index = self.snp_location([function, internal_program, [1]], program)

                if not flag:
                    program = program[:index] + [function, *internal_program, [p]] + program[index:]
                    depth += (internal_depth + 1)*abs(p)
                    #program = program[:index] + [function, *internal_program, [1]] + program[index:]
                    #depth += internal_depth + 2
                    single_fun = False
                    
            else:
                
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                    
                flag, index = self.snp_location([terminal], program)
                
                if not flag:
                    program = program[:index] + [terminal] + program[index:]
                    depth +=1
                    single_fun = False

                
        
        #program.sort(key=lambda p: p[0].name) 
        
        #print("sum")
        #self.print_program(program)
        if single_fun:
            return program, depth
        
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp)    
            if p == 0:
                p = 1
        return [self.sum, *program, [p]], depth*abs(p)
        return [self.sum, *program, [1]], depth
        
    def program_times(self, random_state, init_depth=0):
        
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        
        function_list = list(range(len(self.function_set)))
        
        feature_list = list(range(self.n_features))

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(function_list))
        function = function_list[function]
        #function_list.remove(function)
        function = self.function_set[function]
        var = random_state.randint(self.n_features)
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp) 
            if p == 0:
                p = 1
        program = [function, var,[p]]
        #program = [function, var,[1]]
        #depth = max(3, init_depth)
        depth = max(2*abs(p), init_depth)
        
        single_fun = True
        
        while depth < max_depth:
            n_choice = len(feature_list) + len(function_list) + 1
            #n_choice = len(function_list) + 1
            choice = random_state.randint(n_choice)
            # Determine if we are adding a function or terminal
            # if method == 'full' or choice < len(function_list):
            if choice < len(function_list):
                function = random_state.randint(len(function_list))
                function = function_list[function]
                #function_list.remove(function)
                function = self.function_set[function]
                var = random_state.randint(self.n_features)
                
                p = 1
                if random_state.uniform() < 0.025:
                    p = random_state.randint(self.min_exp, self.max_exp)
                    if p == 0:
            	         p = 1
                                
                flag, index = self.snp_location([function, var], program, True)
                
               
                if flag:
                    ip = program[index+2][0]
                    program[index+2][0] += p
                    if program[index+2][0] > self.max_exp or program[index+2][0] < self.min_exp:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	             p = 1
                        program[index+2][0] = p
                        depth += 2*(abs(p) - ip)
                    else:
                        depth += 2*p
                    #program[index+2][0] += 1
                    #depth += 1
                    
                else:
                    program = program[:index] + [function, var, [p]] + program[index:]
                    #depth += 2*p + 1
                    #program = program[:index] + [function, var, [1]] + program[index:]
                    #depth += 3
                    single_fun = False
                    depth += abs(2*p)
                    
            elif choice == n_choice - 1:
                function = random_state.randint(len(self.function_set))
                function = function_list[function]
                #function_list.remove(function)
                function = self.function_set[function]
                internal_program, internal_depth = self.program_sum(random_state, depth)
                
                p = 1
                if random_state.uniform() < 0.025:
                    p = random_state.randint(self.min_exp, self.max_exp)
                    if p == 0:
            	         p = 1
                
                flag, index = self.snp_location([function, *internal_program], program, True)
                

                
                if flag:
                    ip = program[index+len(internal_program)+1][0]
                    program[index+len(internal_program)+1][0] += p
                    if program[index+len(internal_program)+1][0] > self.max_exp or program[index+len(internal_program)+1][0] < self.min_exp:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	             p = 1
                        program[index+len(internal_program)+1][0] = p
                        depth += (internal_depth+1)*(abs(p)- ip)
                    else:
                        depth += (internal_depth+1)*abs(p)
                    #program[index+len(internal_program)+1][0] += 1
                    #depth += internal_depth + 2
                else:
                    program = program[:index] + [function, *internal_program, [p]] + program[index:]
                    #depth += (internal_depth + 1)*p + 1
                    #program = program[:index] + [function, *internal_program, [1]] + program[index:]
                    #depth += internal_depth + 2
                    depth += (internal_depth+1)*abs(p)
                    single_fun = False
            else:
                
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                
                flag, index = self.snp_location([terminal], program)
                
                if not flag: 
                    program = program[:index] + [terminal] + program[index:]                  
                    depth += 1
                    single_fun = False
       

        
        if single_fun:
            return program, depth
        
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp) 
            if p == 0:
                p = 1
        
        return [self.prod, *program, [p]], depth*abs(p)    
        return [self.prod, *program, [1]], depth
        

    def build_program_with_prior(self, random_state, init_depth=0):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
        
        prior : An array with prior probabilities for each
            function in the function set.
        
        eps : The cut off probability to exlude functions.
            
        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """        
        sumprod = random_state.uniform()
        if sumprod < 0.5:
            program, depth = self.program_sum_with_prior(random_state, init_depth)
        else:
            program, depth = self.program_times_with_prior(random_state, init_depth)       
        
        p0 = self.p0_init(program, random_state)
        
        #self.print_program(program)
        
        return program, p0

    def program_sum_with_prior(self, random_state, init_depth=0):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        
        function_list = list(range(len(self.function_set)))
        
        feature_list = list(range(self.n_features))

        # Start a program with a function to avoid degenerative programs
        function = np.searchsorted(self.CDF, random_state.uniform())
        #print(function)
        function = self.prior_functions[function]
        var = random_state.randint(self.n_features)
        
        if isinstance(function, _Function):
            p = 1
            if random_state.uniform() < 0.025:
                p = random_state.randint(self.min_exp, self.max_exp)
                if p == 0:
                    p = 1
        else:
            p = function
            function = self.affine
            
        program = [function, var, [p]]
        depth = max(2*abs(p), init_depth)
        #program = [function, var, [1]]
        #depth = max(3, init_depth)
        
        single_fun = True
        
        while depth < max_depth:
            n_choice = len(feature_list) + len(function_list) + 2
            #n_choice = len(function_list) + 2
            choice = random_state.randint(n_choice)
            #print(n_choice, choice, len(function_list))
            # Determine if we are adding a function or terminal
            # if method == 'full' or choice < len(function_list):
            if choice < len(function_list):
                function = np.searchsorted(self.CDF, random_state.uniform())
                #print(function)
                function = self.prior_functions[function]
                var = random_state.randint(self.n_features)
                if isinstance(function, _Function):
                    p = 1
                    if random_state.uniform() < 0.025:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	            p = 1
                else:
            	    p = function
            	    function = self.affine
            	    
                flag, index = self.snp_location([function, var, [p]], program)
                #flag, index = self.snp_location([function, var, [1]], program)
                
                if not flag:
                    program = program[:index] + [function, var, [p]] + program[index:]
                    depth += 2*abs(p)
                    #program = program[:index] + [function, var, [1]] + program[index:]
                    #depth += 3
                    single_fun = False

            elif choice == n_choice - 2:
                internal_program, internal_depth = self.program_times_with_prior(random_state, depth)
                flag, index = self.snp_location(internal_program, program)
                #if internal_program not in program:
                if not flag:
                    program = program[:index] + internal_program + program[index:]                    
                    depth += internal_depth
                    single_fun = False
                                    
            elif choice == n_choice - 1:
                function = np.searchsorted(self.CDF, random_state.uniform())
                function = self.prior_functions[function]
                if isinstance(function, _Function):
                    p = 1
                    if random_state.uniform() < 0.025:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	            p = 1
                else:
            	    p = function
            	    function = self.affine
                internal_program, internal_depth = self.program_sum_with_prior(random_state, depth)
                
                flag, index = self.snp_location([function, internal_program, [p]], program)
                #flag, index = self.snp_location([function, internal_program, [1]], program)

                if not flag:
                    program = program[:index] + [function, *internal_program, [p]] + program[index:]
                    depth += (internal_depth + 1)*abs(p)
                    #program = program[:index] + [function, *internal_program, [1]] + program[index:]
                    #depth += internal_depth + 2
                    single_fun = False
                    
            else:
                
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                    
                flag, index = self.snp_location([terminal], program)
                
                if not flag:
                    program = program[:index] + [terminal] + program[index:]
                    depth +=1
                    single_fun = False

                
        
        #program.sort(key=lambda p: p[0].name) 
        
        #print("sum")
        #self.print_program(program)
        if single_fun:
            return program, depth
        
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp)    
            if p == 0:
                p = 1
        return [self.sum, *program, [p]], depth*abs(p)
                
    def program_times_with_prior(self, random_state, init_depth=0):

        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
        
        function_list = list(range(len(self.function_set)))
        
        feature_list = list(range(self.n_features))

        # Start a program with a function to avoid degenerative programs
        function = np.searchsorted(self.CDF, random_state.uniform())
        #print(function)
        function = self.prior_functions[function]
        var = random_state.randint(self.n_features)
        
        if isinstance(function, _Function):
            p = 1
            if random_state.uniform() < 0.025:
                p = random_state.randint(self.min_exp, self.max_exp)
                if p == 0:
                    p = 1
        else:
            p = function
            function = self.affine
            
        program = [function, var,[p]]
        #program = [function, var,[1]]
        #depth = max(3, init_depth)
        depth = max(2*abs(p), init_depth)
        
        single_fun = True
       
        while depth < max_depth:
            n_choice = len(feature_list) + len(function_list) + 1
            #n_choice = len(function_list) + 1
            choice = random_state.randint(n_choice)
            # Determine if we are adding a function or terminal
            # if method == 'full' or choice < len(function_list):
            if choice < len(function_list):
                function = np.searchsorted(self.CDF, random_state.uniform())
                #print(function)
                function = self.prior_functions[function]
                var = random_state.randint(self.n_features)
                if isinstance(function, _Function):
                    p = 1
                    if random_state.uniform() < 0.025:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	            p = 1
                else:
            	    p = function
            	    function = self.affine
                                
                flag, index = self.snp_location([function, var], program, True)
                
               
                if flag:
                    ip = program[index+2][0]
                    program[index+2][0] += p
                    if program[index+2][0] > self.max_exp or program[index+2][0] < self.min_exp:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	             p = 1
                        program[index+2][0] = p
                        depth += 2*(abs(p) - ip)
                    else:
                        depth += 2*p
                    #program[index+2][0] += 1
                    #depth += 1
                    
                else:
                    program = program[:index] + [function, var, [p]] + program[index:]
                    #depth += 2*p + 1
                    #program = program[:index] + [function, var, [1]] + program[index:]
                    #depth += 3
                    single_fun = False
                    depth += abs(2*p)
                    
            elif choice == n_choice - 1:
                function = np.searchsorted(self.CDF, random_state.uniform())
                function = self.prior_functions[function]
                
                if isinstance(function, _Function):
                    p = 1
                    if random_state.uniform() < 0.025:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	            p = 1
                else:
            	    p = function
            	    function = self.affine
            	    
                internal_program, internal_depth = self.program_sum_with_prior(random_state, depth)
                
                flag, index = self.snp_location([function, *internal_program], program, True)
                

                
                if flag:
                    ip = program[index+len(internal_program)+1][0]
                    program[index+len(internal_program)+1][0] += p
                    if program[index+len(internal_program)+1][0] > self.max_exp or program[index+len(internal_program)+1][0] < self.min_exp:
                        p = random_state.randint(self.min_exp, self.max_exp)
                        if p == 0:
            	             p = 1
                        program[index+len(internal_program)+1][0] = p
                        depth += (internal_depth+1)*(abs(p)- ip)
                    else:
                        depth += (internal_depth+1)*abs(p)
                    #program[index+len(internal_program)+1][0] += 1
                    #depth += internal_depth + 2
                else:
                    program = program[:index] + [function, *internal_program, [p]] + program[index:]
                    #depth += (internal_depth + 1)*p + 1
                    #program = program[:index] + [function, *internal_program, [1]] + program[index:]
                    #depth += internal_depth + 2
                    depth += (internal_depth+1)*abs(p)
                    single_fun = False
            else:
                
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                
                flag, index = self.snp_location([terminal], program)
                
                if not flag: 
                    program = program[:index] + [terminal] + program[index:]                  
                    depth += 1
                    single_fun = False
       

        
        if single_fun:
            return program, depth
        
        p = 1
        if random_state.uniform() < 0.025:
            p = random_state.randint(self.min_exp, self.max_exp) 
            if p == 0:
                p = 1
        
        return [self.prod, *program, [p]], depth*abs(p)    


    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        
        if len(self.p0) != len(self.program):
            print("p0 != program")
            return False
            
        balance = 0
        i = 0
        last = 0
        for node in self.program:
            if isinstance(node, _Function):
                balance += 1
                last = i
            elif isinstance(node, list):
                balance -= 1
                if i == last + 1:
                    print("i == last + 1")
                    self.print_program(pflag=False)
                    return False
            i += 1
        
        if balance != 0:
            print("balance != 0")
            self.print_program(self.program)        
        return balance == 0

    def __str__(self, program=None, p0=None, pflag=True):

        if program is None:
            program = self.program

        if p0 is None:
            p0 = self.p0

        stack = []
        if pflag:
            for (p, n) in zip(p0, program):
                if isinstance(n, _Function):
                    if p != 1.0:
                        stack.append(p)
                    stack.append(n.name)
                elif isinstance(n, int):
                    if p != 1.0:
                        stack.append(p)
                    stack.append(n)
                else:
                    stack.append(n)
        else:
            for n in program:
                if isinstance(n, _Function):
                    stack.append(n.name)
                elif isinstance(n, int):
                    stack.append(n)
                else:
                    stack.append(n)

        return stack

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        program_expanded = self.expand(self.program)
        terminals = [0]
        depth = 1
        for node in program_expanded:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""

        stack = [0]
        for node in self.program:
            if isinstance(node, list):
               length = stack.pop() + 1
               length *= abs(node[0])
               if len(stack) == 0:
                   self.print_program(self.program)
               stack[-1] += length
            elif isinstance(node, int) or isinstance(node, float):
               stack[-1] += 1
            else:
               stack.append(0)
        
        if len(stack) > 1:
           print("stack size")
           print(stack)        
        return stack[-1]

    def p0_init(self, program, random_state):
    
        if program is None:
            program = self.program
            
        p0 = []
        for node in program:
            if isinstance(node, _Function) or isinstance(node, int):
                if random_state.uniform() < 0.1:
                    p0.append(random_state.uniform(*self.const_range))
                else:
                    p0.append(1.0)
            else:
                p0.append(1.0)
                    
        return p0
    
    def fit_program(self, exe, X, y, p0=None, sigma=None, **kw):
    
        popt, pcov = curve_fit(exe, X, y, p0, sigma, **kw)
        
        if sigma is None:
            chi2 = sum(((exe(X,*popt)-y))**2)
        else:
            chi2 = sum(((exe(X,*popt)-y)/sigma)**2)
        dof = len(y) - len(popt)
        rchi2 = chi2/dof
        #print('results of general_fit:')
        #print('   chi squared = ', chi2)
        #print('   degrees of freedom = ', dof)
        #print('   reduced chi squared = ', rchi2)

        # The uncertainties are the square roots of the diagonal elements
        punc = np.zeros(len(popt))
        for i in range(0,len(popt)):
            punc[i] = np.sqrt(pcov[i,i])
        return popt, punc, rchi2, dof
    
    def execute1(self, *args):
        
        for i in range(len(self.fit_index)):
            index = self.fit_index[i]
            self.p0[index] = args[1+i]
            
        return self.execute(args[0])
            
    def execute(self, X, pflag=False):
        
        index = -1
        p0index = -1
        
        
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]
        #self.print_program() 
        self.truncate = []    
        apply_stack = []
        for (p0, node) in zip(self.p0, self.program):
            index += 1
            if isinstance(node, list):
                function = apply_stack[-1][2]
                func_test = function
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][3:]]
                if function in [self.sum, self.prod]:
                    intermediate_result = function(terminals)
                else:
                    intermediate_result = function(*terminals)
                #if node[0] != 1 and function.name != 'exp': 
                if node[0] != 1:
                    #print(node[0])
                    function = self.mul
                    terminals = intermediate_result
                    #print(intermediate_result)
                    i = node[0]
                    i = abs(node[0])
                    intermediate_result = np.ones(len(terminals))
                    while i > 0:
                        #print(max(intermediate_result))
                        intermediate_result = function(intermediate_result, terminals)
                        i -= 1
                    if node[0] < 0:
                        intermediate_result[np.where(intermediate_result == 0.0)] = 1e-15 
                        intermediate_result = 1.0 / intermediate_result
                    if False:#any(np.isnan(intermediate_result)):
                        self.print_program(self.program)
                        print(func_test.name, node[0], index)
                        print()
                    #intermediate_result = function(terminals, node[0])    
                    #print(intermediate_result)
                    #print()
                #elif node[0] < 0 and function.name == 'exp':
                #    intermediate_result = 1.0 / intermediate_result

                intermediate_result *= apply_stack[-1][1]
                
                if abs(max(intermediate_result) - min(intermediate_result)) < 1.0e-4:
                    self.truncate.append([apply_stack[-1][0], index+1, (max(intermediate_result) + min(intermediate_result))/2.0]) 
                     
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    #print(X)
                    #print(intermediate_result)
                    #if pflag and len(truncate) > 0:
                    #if len(truncate) > 0:
                        #print(min(intermediate_result), max(intermediate_result))
                        #print(truncate)
                        #self.print_program(pflag=False)
                        #self.truncate_program(truncate)
                        #print()
                        #self.print_program(pflag=False)
                        #print()
                    return intermediate_result
            elif isinstance(node, int):
                
                coef = np.repeat(p0, X.shape[0])
                terminals = X[:, node]
                intermediate_result = self.mul(coef, terminals)
                apply_stack[-1].append(intermediate_result)
                
                
                #apply_stack[-1].append(node)
                #print("int", node)
            elif isinstance(node, float):
                apply_stack[-1].append(node)
            else:
                
                apply_stack.append([index, p0, node])
                #apply_stack.append([p0[p0index], node])
                
                #apply_stack.append([node])
                #print("fun", node.name)

        # We should never get here
        return None
                
    def truncate_program(self, truncate=None):
    
        final_truncate = []
        
        if truncate is None:
            truncate = self.truncate
                      
        for t0 in truncate:
            flag = True
            for t1 in truncate: 
                if t0[0] > t1[0] and t0[1] < t1[1]:
                    flag = False
                    break
            if flag:
                index = len(final_truncate)
                for i in range(len(final_truncate)):
                    t2 = final_truncate[i]
                    if t2[0] > t0[0]:
                        index = i
                final_truncate.insert(index, t0)
        
        l = 0  
        #print("truncate:", len(self.program))  
        for t in final_truncate:
            self.program = self.program[:t[0]-l] + [t[2]] + self.program[t[1]-l:]
            self.p0 = self.p0[:t[0]-l] + [1.0] + self.p0[t[1]-l:]
            l += t[1] - t[0] - 1
        #print(len(self.program))
        #print()    
        return None    
        
    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]
    
    def cal_p_ij(self):
        self.p_ij = {}
        total = 0.0
        stack = []
        #for node in self.program:
        #    if isinstance(node, _Function):
        #        if node.name in self.p_ij:
        #            self.p_ij[node.name] += 1
        #        else:
        #            self.p_ij[node.name] = 1
        #        total += 1
        #if total == 0.0:
        #    total = 1
        #for p in self.p_ij:
        #    self.p_ij[p] /= total
        #return self.p_ij

        for node in self.program:
            if isinstance(node, _Function):
                stack.append([node.name])
            elif isinstance(node, list):
                f = stack.pop()
                
                if node[0] != 0:
                    if len(stack) == 0:
                        if len(f) > 1:
                            for fs in f[1:]:
                                #print(f)
                                #print(fs)
                                if fs[0] in self.p_ij:
                                    self.p_ij[fs[0]] += abs(fs[1]*node[0])
                                else:
                                    self.p_ij[fs[0]] = abs(fs[1]*node[0])
                                total += abs(fs[1]*node[0])
                        else:
                            if f[0] in self.p_ij:
                                self.p_ij[f[0]] += abs(node[0])
                            else:
                                self.p_ij[f[0]] = abs(node[0])
                            total += abs(node[0])    
                    else:
                        stack[-1].append([f[0], node[0]])   
                
#                f = stack.pop()
#                total += abs(node[0])
#                if f in self.p_ij:
#                    self.p_ij[f] += abs(node[0])
#                else:
#                    self.p_ij[f] = abs(node[0])
#            elif isinstance(node, int):
#                total += 1
#                if node in self.p_ij:
#                    self.p_ij[node] += 1
#                else:
#                    self.p_ij[node] = 1
        if total == 0:
            total = 1
        for p in self.p_ij:
            self.p_ij[p] /= total
        
        return self.p_ij
        
                    
    def cal_H_j(self):
        
        self.H_j = 0.0
        for f in self.p_ij:
            self.H_j -= self.p_ij[f] * np.log(self.p_ij[f])
        
        return self.H_j
             
    def cal_H_Rj(self, p_i):
    
        self.H_Rj = 0.0
        for f in self.p_ij:
            self.H_Rj -= self.p_ij[f] * np.log(p_i[f])
        
        return self.H_Rj
            
    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        if True:
            index = []
        
            for i in range(len(self.program)):
                if isinstance(self.program[i], int) or isinstance(self.program[i], _Function):
                    index.append(i)
        
        
            selected_index = np.where(np.random.uniform(size=len(index)) <
                          .15)[0]
            #selected_index = np.where(np.random.uniform(size=len(index)) <
            #                  .25)[0]
        
        
            self.fit_index = []
        
            for i in selected_index:
                self.fit_index.append(index[i])
       
         
            p0 = []
            for i in self.fit_index:
                p0.append(self.p0[i])
            
        
    
            if len(p0) > 0:
                try:    
                    #popt, punc, rc, d = self.fit_program(self.execute1, X, y, p0, bounds=self.const_range, maxfev=10)
                    popt, punc, rc, d = self.fit_program(self.execute1, X, y, p0, bounds=self.const_range, maxfev=5)
                    for i in range(len(self.fit_index)):
                        index = self.fit_index[i]
                        self.p0[index] = popt[i]
                except RuntimeError: 
                    i = 0      
        
        #if len(self.truncate) > 0:
        #    self.truncate_program() 
        #y_pred = self.execute(X, self.p0)    
        
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        
        raw_fitness = self.metric(y, y_pred, sample_weight)
        
        self.cal_p_ij()

        return raw_fitness


    def fitness(self, parsimony_coefficient=None, p_i=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        H_Rj = 0.0
        H_j = 0.0
        if not p_i is None:
            H_Rj = self.cal_H_Rj(p_i)
            H_j = self.cal_H_j()
        
        D_j = H_Rj - H_j
        #D_j = 0.0
           
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        
        #penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        penalty = parsimony_coefficient * (self._length() - D_j) * self.metric.sign
        #penalty = parsimony_coefficient * (self._length() + len(self.program) - D_j) * self.metric.sign
        #penalty = parsimony_coefficient * (self._length() + len(self.program)) * self.metric.sign
        #penalty = parsimony_coefficient * self._length() * self.metric.sign
        #penalty = parsimony_coefficient * (len(self.program) - D_j) * self.metric.sign
        
        return self.raw_fitness_ - penalty

    def get_parent(self, node, program=None):
        
        if program is None:
            program = self.program
            
        stack = []   
        for i in range(node):
            if isinstance(program[i], _Function):
                stack.append(i)
            elif isinstance(program[i], list):
                stack.pop()
                 
        if len(stack) > 0:
            parent = stack.pop()
        else:
            parent = 0
             
        return parent
        
    def get_end(self, node, program=None):
        
        if program is None:
            program = self.program
        
        if node == len(program) or isinstance(program[node], list):
            return node + 1
                    
        balance = 0
        
        if isinstance(program[node], _Function): 
            balance = 1
            
        end = node + 1    
        while end < len(program):
            if balance == 0: break
            node = program[end]
            if isinstance(node, _Function):
                balance += 1
            elif isinstance(node, list):
                balance -= 1
            end += 1

        return end
        
    def get_parent_end(self, node, program=None):
        
        parent = self.get_parent(node, program)
        end = self.get_end(node, program)
        
        return parent, end
                                
    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        #probs = np.array([0.9 if isinstance(node, _Function) elif isinstance(node, list) 0.1 else 0.1
        #                  for node in program])
        
        probs = np.zeros(len(program))
        
        for i in range(len(program)):
            if isinstance(program[i], _Function):
                probs[i] = 0.825
            elif isinstance(program[i], list):
                probs[i] = 0.025 
            else:
                probs[i] = 0.15                 

        probs = np.cumsum(probs / probs.sum())
        
        
        start = np.searchsorted(probs, random_state.uniform())
        
        parent, end = self.get_parent_end(start, program)
        
        return start, end, parent

    def reproduce(self):
        """Return a copy of the embedded program."""
        
        program = []
        for p in self.program:
            if isinstance(p, _Function):
                program.append(p)
            elif isinstance(p, list):
                program.append([p[0]])
            else:
                program.append(p)
                    
        return program, copy(self.p0)        

    def crossover(self, donor, p0_donor, random_state, pflag=False):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end, parent = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end, donor_parent = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        
        #print(start, end)
        #self.print_program()
        #program = []
        #for i in range(start):
        #    program.append(self.program[i])
        #for i in range(end+1, len(self.program)):
        #    program.append(self.program[i])
        program = self.program[:start] + self.program[end:]
        p0 = self.p0[:start] + self.p0[end:]
        parent_end = self.get_end(parent, program)
        
        donation = donor[donor_start:donor_end]
        p0_donation = p0_donor[donor_start:donor_end]
        
        if len(program) == 0:
            #print("len, if")
            #print(donor_start, donor_end)
            #self.print_program(donor)
            #self.print_program(donation)
            if isinstance(donation[0], list):
                return [0.0], [1.0], removed, donor_removed
            return donation, p0_donation, removed, donor_removed
        elif len(donation) == 0:
            #print("len, else")
            #self.print_program(program)
            return program, p0, removed, donor_removed
       
        if pflag:
            print(start, end, parent)
            print(donor_start, donor_end, donor_parent)
            self.print_program(program, pflag=False)
            self.print_program(donor, pflag=False)
                        
        if not(isinstance(donor[donor_start], list) or isinstance(self.program[start], list)):
            if self.program[parent] == self.prod:
                if donation[0] == self.prod:
                
                    program_seg = program[parent+1:parent_end-1]
                    p0_seg = p0[parent+1:parent_end-1]
                    
                    #print("prod, prod")
                    #print(start, end)
                    #self.print_program(self.program)
                    #print(parent, parent_end)
                    #self.print_program(program)
                    #print(donor_start, donor_end)
                    #self.print_program(donor)
                    #self.print_program(donation)
                    
                    i = 1
                    while i < len(donation)-1:
                        snp = []
                        snp.append(donation[i])
                        p0_snp = []
                        p0_snp.append(p0_donation[i])
                        if isinstance(donation[i], _Function):
                            balance = 1
                        else:
                            balance = 0
                        while balance > 0:
                            i += 1
                            snp.append(donation[i])
                            p0_snp.append(p0_donation[i])
                            if isinstance(donation[i], _Function):
                                balance += 1
                            elif isinstance(donation[i], list):
                                balance -= 1
                        
                        #self.print_program(snp)
                        if isinstance(snp[-1], list):
                            snp[-1][0] *= donation[-1][0]
                            if snp[-1][0] > self.max_exp or snp[-1][0] < self.min_exp:
                                snp[-1][0] = random_state.randint(self.min_exp, self.max_exp)
                        
                        #self.print_program(program_seg)
                        
                        #flag, index = self.snp_location(snp, program_seg)
                        if isinstance(snp[0], _Function): 
                            flag, index = self.snp_location(snp[:-1], program_seg, True)
                        else:
                            flag, index = self.snp_location(snp, program_seg)
                        #print(flag, index)
                        
                        if not flag:
                            program_seg = program_seg[:index] + snp + program_seg[index:]
                            p0_seg = p0_seg[:index] + p0_snp + p0_seg[index:]
                        else:
                            if isinstance(snp[0], _Function):
                                program_seg[index+len(snp)-1][0] += snp[-1][0]
                                if program_seg[index+len(snp)-1][0] > self.max_exp or program_seg[index+len(snp)-1][0] < self.min_exp:
                                    program_seg[index+len(snp)-1][0] = random_state.randint(self.min_exp, self.max_exp)
                        i += 1
                        
                        #self.print_program(program_seg)
                        
                    program = program[:parent+1] + program_seg + program[parent_end-1:]
                    p0 = p0[:parent+1] + p0_seg + p0[parent_end-1:]
                    
                    #self.print_program(program)
                    return program, p0, removed, donor_removed
                       
                elif donation[0] == self.sum:
                    
                    program_seg = program[parent+1:parent_end-1]
                    p0_seg = p0[parent+1:parent_end-1]
                    
                    #print("prod, sum")
                    #print(start, end)
                    #self.print_program(self.program)
                    #print(parent, parent_end)
                    #self.print_program(program)
                    #print(donor_start, donor_end)
                    #self.print_program(donor)
                    #self.print_program(donation)
                    
                    #program_seg[0] = self.sum
                    #program_seg[-1][0] = [1]
                    
                    snp_total = []
                    p0_snp_total = []
                    i = 1
                    while i < len(donation)-1:
                        snp = []
                        snp.append(donation[i])
                        p0_snp = []
                        p0_snp.append(p0_donation[i])
                        if isinstance(donation[i], _Function):
                            balance = 1
                        else:
                            balance = 0
                        while balance > 0:
                            i += 1
                            snp.append(donation[i])
                            p0_snp.append(p0_donation[i])
                            if isinstance(donation[i], _Function):
                                balance += 1
                            elif isinstance(donation[i], list):
                                balance -= 1
                        
                        if pflag:
                            self.print_program(snp, pflag=False)
                            self.print_program(program_seg, pflag=False)
                        
                        if isinstance(snp[0], _Function):
                            flag, index = self.snp_location(snp[:-1], program_seg, True)
                        else:
                            flag, index = self.snp_location(snp, program_seg)
                        
                        #print(flag, index)
                        
                        if not flag:
                            snp = program_seg[:index] + snp + program_seg[index:]
                            p0_snp = p0_seg[:index] + p0_snp + p0_seg[index:]
                        else:
                            if isinstance(snp[0], _Function):
                                tmp = []
                                for n in program_seg:
                                    if isinstance(n, _Function):
                                        tmp.append(n)
                                    elif isinstance(n, list):
                                        tmp.append([n[0]])
                                    else:
                                        tmp.append(n) 
                                
                                tmp[index+len(snp)-1][0] += snp[-1][0]
                                if tmp[index+len(snp)-1][0] > self.max_exp or tmp[index+len(snp)-1][0] < self.min_exp:
                                    tmp[index+len(snp)-1][0] = random_state.randint(self.min_exp, self.max_exp)
                                snp = tmp
                            else:
                                snp = program_seg
                            
                            p0_snp = p0_seg
                            
                        snp = [self.prod] + snp + [[1]] 
                        p0_snp = [1.0] + p0_snp + [1]  
                        
                        if pflag:
                            self.print_program(snp, pflag=False)  
                            print()
                        
                        flag, index = self.snp_location(snp, snp_total)
                        
                        
                        #print(flag, index)
                        
                        if not flag:
                            snp_total = snp_total[:index] + snp + snp_total[index:]
                            p0_snp_total = p0_snp_total[:index] + p0_snp + p0_snp_total[index:]
                        
                        if pflag:
                            self.print_program(snp_total, pflag=False)
                        
                        i += 1
                        
                        
                    snp_total.insert(0, self.sum)
                    snp_total.append([1])
                    
                    p0_snp_total.insert(0, 1.0)
                    p0_snp_total.append(1.0)
                    
                    program = program[:parent] + snp_total + program[parent_end:]
                    p0 = p0[:parent] + p0_snp_total + p0[parent_end:]
                    
                    #self.print_program(program)
                        
                    return program, p0, removed, donor_removed
                else:
                    #print("prod, else")
                    #flag, index = snp_location(donor[donor_start:donor_end], self.program[parent, parent_end])
                    #program = self.program[:index] + donor[donor_start:donor_end] + self.program[index:]
                    program = self.program[:start] + donor[donor_start:donor_end] + self.program[end:]
                    p0 = self.p0[:start] + p0_donor[donor_start:donor_end] + self.p0[end:]
                    #self.print_program(program)
                    return program, p0, removed, donor_removed
                
            elif self.program[parent] == self.sum:
                if donation[0] == self.prod:
                    #print("sum, prod")
                    #print(start, end)
                    #self.print_program(self.program)
                    #print(parent, parent_end)
                    #self.print_program(program)
                    #print(donor_start, donor_end)
                    #self.print_program(donor)
                    #self.print_program(donation)
                    flag, index = self.snp_location(donation, program[parent+1:parent_end-1])
                    
                    index += parent + 1
                    
                    if not flag:
                        program = program[:index] + donation + program[index:]
                        p0 = p0[:index] + p0_donation + p0[index:]
                    #self.print_program(program)
                    return program, p0, removed, donor_removed
                elif donation[0] == self.sum:
                
                    program_seg = program[parent+1:parent_end-1]
                    p0_seg = p0[parent+1:parent_end-1]
                    
                    #print("sum, sum")
                    #print(start, end)
                    #self.print_program(self.program)
                    #print(parent, parent_end)
                    #self.print_program(program)
                    #print(donor_start, donor_end)
                    #self.print_program(donor)
                    #self.print_program(donation)
                    
                    i = 1
                    while i < len(donation)-1:
                        snp = []
                        snp.append(donation[i])
                        p0_snp = []
                        p0_snp.append(p0_donation[i])
                        if isinstance(donation[i], _Function):
                            balance = 1
                        else:
                            balance = 0
                        while balance > 0:
                            i += 1
                            snp.append(donation[i])
                            p0_snp.append(p0_donation[i])
                            if isinstance(donation[i], _Function):
                                balance += 1
                            elif isinstance(donation[i], list):
                                balance -= 1
                        
                        #self.print_program(snp)
                        
                        #self.print_program(program_seg)
                        
                        flag, index = self.snp_location(snp, program_seg)
                        
                        if not flag:
                            program_seg = program_seg[:index] + snp + program_seg[index:]
                            p0_seg = p0_seg[:index] + p0_snp + p0_seg[index:]
                        
                        i += 1
                        
                        #self.print_program(program_seg)
                        
                    program = program[:parent+1] + program_seg + program[parent_end-1:]
                    p0 = p0[:parent+1] + p0_seg + p0[parent_end-1:]
                    
                    #self.print_program(program)
                    return program, p0, removed, donor_removed
                else:
                    #print("sum, else")
                    #print(start, end)
                    #self.print_program(self.program)
                    #print(parent, parent_end)
                    #self.print_program(program)
                    #print(donor_start, donor_end)
                    #self.print_program(donor)
                    #self.print_program(donation)

                    flag, index = self.snp_location(donation, program[parent+1:parent_end-1])
                    
                    index += parent + 1
                    
                    #print(flag, index)
                    #self.print_program(program[parent+1:parent_end-1])
                    
                    if not flag:
                        program = program[:index] + donation + program[index:]
                        p0 = p0[:index] + p0_donation + p0[index:]
                        
                    #self.print_program(program)
                    
                    return program, p0, removed, donor_removed
            
            else:
                #print("else")
                program = self.program[:start] + donation + self.program[end:]
                p0 = self.p0[:start] + p0_donation + self.p0[end:]
                #self.print_program(program)
                return program, p0, removed, donor_removed
        else:
            return self.program, self.p0, [], []


    def subtree_mutation(self, random_state, prior=None):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        if prior:
            chicken, p0 = self.build_program_with_prior(random_state)
        else:
            chicken, p0 = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, p0, random_state)   

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end, parent = self.get_subtree(random_state)

        subtree = self.program[start:end]
        p0_subtree = self.p0[start:end]

        # Get a subtree of the subtree to hoist
        sub_start, sub_end, sub_parent = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        p0_hoist = p0_subtree[sub_start:sub_end]
        
        
            
        if False:
            print("hoist")
            self.print_program()
            print(start, end, parent)
            self.print_program(subtree)
            print(sub_start, sub_end, sub_parent)
            self.print_program(hoist)
            if (isinstance(hoist[0], list) or isinstance(self.program[start], list)):
                self.print_program()
            else:
                self.print_program(self.program[:start] + hoist + self.program[end:])
        
            print()
        
        
        
        
        if len(hoist) < len(subtree) and not(isinstance(hoist[0], list) or isinstance(self.program[start], list)):
        
            if self.program[parent] in [self.prod, self.sum]:
                program = self.program[:start] + self.program[end:]
                p0 = self.p0[:start] + self.p0[end:]
                parent_end = self.get_end(parent, program)
                flag, index = self.snp_location(hoist, program[parent+1:parent_end-1])
                
                if False:
                    self.print_program(program)
                    print(parent_end)
                    self.print_program(program[parent+1:parent_end-1])
                    print(index)
                    
                if not flag:
                    # Determine which nodes were removed for plotting
                    removed = list(set(range(start, end)) -
                                   set(range(start + sub_start, start + sub_end)))
                                   
                    program = program[:parent+index+1] + hoist + program[parent+index+1:]
                    p0 = p0[:parent+index+1] + p0_hoist + p0[parent+index+1:]
                    #self.print_program(program)
                    return program, p0, removed
                else:
                    #print("else not flag")
                    removed = list(set(range(start, end)) -
                                   set(range(start + sub_start, start + sub_end)))
                                   
                    program = program[:parent+index+1] + program[parent+index+1:]
                    p0 = p0[:parent+index+1] + p0[parent+index+1:]
                    #self.print_program(program)
                    return program, p0, removed
                
            else:
            
                removed = list(set(range(start, end)) -
                               set(range(start + sub_start, start + sub_end)))
                program = self.program[:start] + hoist + self.program[end:]
                p0 = self.p0[:start] + p0_hoist + self.p0[end:]
                #self.print_program(program)
                return program, p0, removed
        else:
            return self.program, self.p0, []
                
        print()
        print()

        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
                       
          
        if (isinstance(hoist[0], list) or isinstance(self.program[start], list)):
            return  self.program, []
            
        return self.program[:start] + hoist + self.program[end:], removed
    
    def point_mutation(self, random_state, prior=None, pflag=False):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        #pflag = False
        #program = copy(self.program)
        program = []
        for p in self.program:
            if isinstance(p, _Function):
                program.append(p)
            elif isinstance(p, list):
                program.append([p[0]])
            else:
                program.append(p)

        p0 = copy(self.p0)
        # Get the nodes to modify
        #print("point")
        #self.print_program(program)
                          
        probs = np.zeros(2*len(program))
        for i in range(len(program)):
            if isinstance(program[i], _Function):
                probs[i] = 0.825
                probs[i+len(program)] = 0.15
            elif isinstance(program[i], list):
                probs[i] = 0.025 
            elif isinstance(program[i], int):
                probs[i] = 0.15
                probs[i+len(program)] = 0.15
            else:
                probs[i] = 0.15                 

        probs = np.cumsum(probs / probs.sum())
        
        n_mutate = len(np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0])
        
        mutate = []
        mutate_p0 = []
        for i in range(n_mutate):
            m = np.searchsorted(probs, random_state.uniform())
            while m in mutate:
                m = np.searchsorted(probs, random_state.uniform())
            if m < len(program):
                mutate.append(m)
            else:
                mutate_p0.append(m-len(program))
                
        for node in mutate_p0:
            p0[node] += random_state.uniform(-1.0, 1.0)
            #p0[node] = random_state.uniform(*self.const_range)
            if p0[node] > self.const_range[1] or p0[node] < self.const_range[0]:
                p0[node] = random_state.uniform(*self.const_range)
        
        mutate = sorted(mutate)
        
        if pflag:
            self.print_program(program, pflag=False)
            print(mutate)
            print()

        for node in mutate:   
            if node >= len(program):
                continue                     
            if pflag:
                self.print_program(program, pflag=False)
                print(node)
                print()
            if isinstance(program[node], _Function):
                parent = self.get_parent(node, program)
                
                if program[node] == self.sum:
                    end = self.get_end(node, program)
                    if program[parent] == self.prod:
                        #self.print_program(program)
                        program.pop(node)
                        program.pop(end-2)
                        p0.pop(node)
                        p0.pop(end-2)
                        #self.print_program(program)
                        #print()
                    else:
                        program_seg = program[node+1:end-1]
                        p0_seg = p0[node+1:end-1]
                        if pflag:
                            self.print_program(program, pflag=False)
                            print(node, end)
                            self.print_program(program_seg, pflag=False)
                        
                        snp_total = []
                        p0_snp_total = []
                        i = 0
                        
                        if len(program_seg) == 1:
                            snp_total = program_seg
                            p0_snp_total = p0_seg
                            
                        while i < len(program_seg)-1:
                            if program_seg[i] == self.prod:
                                #self.print_program(program_seg)
                                tmp_end = self.get_end(i, program_seg)
                                program_seg.pop(tmp_end-1)
                                p0_seg.pop(tmp_end-1)
                                i += 1
                                #self.print_program(program_seg)
                                #print() 
                                
                            snp = []
                            snp.append(program_seg[i])
                            p0_snp = []
                            p0_snp.append(p0_seg[i])
                            if isinstance(program_seg[i], _Function):
                                balance = 1
                            else:
                                balance = 0
                            while balance > 0:
                                i += 1
                                snp.append(program_seg[i])
                                p0_snp.append(p0_seg[i])
                                if isinstance(program_seg[i], _Function):
                                    balance += 1
                                elif isinstance(program_seg[i], list):
                                    balance -= 1

                            if isinstance(snp[0], _Function):
                                flag, index = self.snp_location(snp[:-1], snp_total, True)
                            else:
                                flag, index = self.snp_location(snp, snp_total)
                            
                            #self.print_program(snp)
                            #self.print_program(snp_total)
                            #print(flag, index)
                            
                            if not flag:
                                snp_total = snp_total[:index] + snp + snp_total[index:]
                                p0_snp_total = p0_snp_total[:index] + p0_snp + p0_snp_total[index:]
                            else:
                                if isinstance(snp[0], _Function):
                                    snp_total[index+len(snp)-1][0] += snp[-1][0]
                                    if snp_total[index+len(snp)-1][0] > self.max_exp or snp_total[index+len(snp)-1][0] < self.min_exp:
                                        snp_total[index+len(snp)-1][0] = random_state.randint(self.min_exp, self.max_exp)
                            i += 1
                        
                        program = program[:node+1] + snp_total + program[end-1:]
                        p0 = p0[:node+1] + p0_snp_total + p0[end-1:]

                        program[node] = self.prod
                        #self.print_program(program)
                        #print()
                        
                elif program[node] == self.prod:
                    if program[parent] != self.prod:
                        program[node] = self.sum
                    #else:
                    #    print("parent")
                else:
                    
                    arity = program[node].arity
                    # Find a valid replacement with same arity
                    replacement_fun = len(self.arities[arity])
                    replacement_fun = random_state.randint(replacement_fun)
                    replacement_fun = self.arities[arity][replacement_fun]

                    if program[parent] in [self.sum, self.prod]:
                        node_end = self.get_end(node, program)
    
                        replacement = program[node:node_end]
                        p0_replacement = p0[node:node_end]
                        replacement[0] = replacement_fun
                        program = program[:node] + program[node_end:]
                        p0 = p0[:node] + p0[node_end:]
                        
                        
                        stack = []
                        stack.append(parent)
                        i = parent + 1
                        index = 0
                        flag = True
                        
                        while len(stack) > 0:
                            if program[parent] == self.prod and program[i:i+len(replacement)-1] == replacement[:-1] and len(stack) == 1:
                                
                                flag = False
                                #print("point prod")
                                #self.print_program(program)
                                #self.print_program(pro_)
                                #self.print_program(replacement)
                                #print(i, pro_[i+len(replacement)-1])
                                #print()
                                
                                
                                #pro_[i+len(replacement)-1][0] += replacement[-1][0]
                                #self.print_program(pro_)
                                #print()
                                break

                            elif program[parent] == self.sum and program[i:i+len(replacement)] == replacement and len(stack) == 1:
                                    
                                flag = False
                                #print("point sum")
                                #self.print_program(program)
                                #self.print_program(pro_)
                                #self.print_program(replacement)
                                #print(i, pro_[i+len(replacement)-1])
                                #print()
                                break
                                
                              
                            
                            if isinstance(program[i], _Function):
                                if program[i].name > replacement[0].name and len(stack) == 1:
                                    index = i
                                stack.append(i)
                            elif isinstance(program[i], list):
                                stack.pop() 
                                    
                            i += 1

                         
                        if index == 0:
                            index = i - 1
                        
                        if flag:
                            program = program[:index] + replacement + program[index:]
                            p0 = p0[:index] + p0_replacement + p0[index:]
                            
                    else:
                        program[node] = replacement_fun  
      
            elif isinstance(program[node], list):
                
                program[node][0] = program[node][0] + random_state.randint(-2, 3)
                
                if program[node][0] > self.max_exp or program[node][0] < self.min_exp:
                    program[node][0] = random_state.randint(self.min_exp, self.max_exp)  
                
                #if abs(program[node][0]) > 20:
                #    program[node][0] = random_state.randint(-20,21)
                
                if program[node][0] == 0:
                    parent = self.get_parent(node, program)
                    #print("zero power", node, parent)
                    #self.print_program(program)
                    program = program[:parent] + [1.0] + program[node+1:]
                    p0 = p0[:parent] + [1.0] + p0[node+1:]
                    #self.print_program(program)
                    #pflag = True
                 
            else:
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                
                program[node] = terminal                    
        
        if pflag:
            self.print_program(program, pflag=False)   
            
        #self.print_program(program)
        #print()
            
        return program, p0, list(mutate)

    def copy_program(program):
        if program == None:
            return None
        copied = []
        for p in program:
            if isinstance(p, _Function):
                copied.append(p)
            elif isinstance(p, list):
                copied.append([p[0]])
            else:
                copied.append(p)
        return copied
        
    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
    
    
