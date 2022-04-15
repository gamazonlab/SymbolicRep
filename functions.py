"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import joblib as jl
#from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity, wrap=False):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=jl.wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def _pow(x, p):
   return x**p
   
def _F_of_functions(f, g):
    def fog(x):
        return f(g(x))
    _FOG = _Function(function=fog, name='_FOG', arity=1)
    return _FOG
    
def _F_of_2functions(f, g, h):
    def fogh(x):
        return f(g(x), h(x))
    _FOGH = _Function(function=fogh, name='_FOGH', arity=1) 
    return _FOGH

def _F_of_function_float(f, g, c):
    def fogc(x):
        return f(g(x), c)
    _FOGC = _Function(function=fogc, name='_FOGC', arity=1) 
    return _FOGC

def _F_of_float_function(f, g, c):
    def focg(x):
        return f(c, g(x))
    _FOCG = _Function(function=focg, name='_FOCG', arity=1) 
    return _FOCG

def _F_of_function_x(f, g):
    def fogx(x):
        return f(g(x), x)
    _FOGX = _Function(function=fogx, name='_FOGX', arity=1) 
    return _FOGX
    
def _F_of_x_function(f, g):
    def foxg(x):
        return f(x, g(x))
    _FOXG = _Function(function=foxg, name='_FOXG', arity=1) 
    return _FOXG
    
def _F_of_x_float(f, c):
    def foxc(x):
        return f(x, c)
    _FOXC = _Function(function=foxc, name='_FOXC', arity=1) 
    return _FOXC
   
def _F_of_float_x(f, c):
    def focx(x):
        return f(c, x)
    _FOCX = _Function(function=focx, name='_FOCX', arity=1) 
    return _FOCX
  
def _F_of_xx(f):
    def fxx(x):
        return f(x, x)
    _FXX = _Function(function=fxx, name='_FXX', arity=1)
    return _FXX
      
def _ode_function(f):
    def of(y, t):
       return f(t)
    return of
   
def _sum(x):
    return np.sum(x, axis=0)
   
def _prod(x):
    return np.prod(x, axis=0)
    
def _affine(x):
    return x
         
add2 = _Function(function=np.add, name='add', arity=2)
sumn = _Function(function=_sum, name='sum', arity=1)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
prodn = _Function(function=_prod, name='prod', arity=1)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)
pow1 = _Function(function=_pow, name='pow', arity=2)
affine = _Function(function=_affine, name='affine', arity=1)
FOG = _Function(function=_F_of_functions, name='FOG', arity=2)
FOXX = _Function(function=_F_of_xx, name='FOXX', arity=1)
FOGH = _Function(function=_F_of_2functions, name='FOGH', arity=3)
FOGC = _Function(function=_F_of_function_float, name='FOGC', arity=3)
FOCG = _Function(function=_F_of_float_function, name='FOCG', arity=3)
FOGX = _Function(function=_F_of_function_x, name='FOGX', arity=2)
FOXG = _Function(function=_F_of_x_function, name='FOXG', arity=2)
FOXC = _Function(function=_F_of_x_float, name='FOXC', arity=2)
FOCX = _Function(function=_F_of_float_x, name='FOCX', arity=2)
ODEF = _Function(function=_ode_function, name='ODEF', arity=1)

_function_map = {'add': add2,
                 'sum': sumn,
                 'sub': sub2,
                 'mul': mul2,
                 'prod': prodn,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'sig': sig1,
                 'pow': pow1,
                 'affine': affine,
                 'fog': FOG,
                 'foxx': FOXX,
                 'fogh': FOGH,
                 'fogc': FOGC,
                 'focg': FOCG,
                 'fogx': FOGX,
                 'foxg': FOXG,
                 'fogc': FOGC,
                 'focg': FOCG,
                 'odef': ODEF}
                 
                 
