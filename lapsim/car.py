import jax.tree_util as jtu
import jax.numpy as jnp
import jax

from dataclasses import dataclass, field, fields
from typing import List


# TODO: type hints


@jax.jit
def expand_like(x, s):
    """
    Expands the input array `x` with length 1 axis at the at the end
    to have the same ndim as `s`.  If `s.ndim` >= `x.ndim` the input
    array will be left unchanged.

    Parameters
    ----------
    x : array_like (has to have `.ndim` property)
        array to expand

    s : array_like
        array to match the ndim

    Returns
    -------
    expanded : array_like
        Array with the same ndim as `s`

    Examples
    --------
    >>> expand_like([0, 1, 0, 1], np.ones([4, 32]))
    [[0, 1, 0, 1]]

    >>> expand_like(1, np.ones([4, 4, 4]))
    [[[1]]]

    >>> expand_like(np.ones([4, 4]), 
                    np.ones([4, 4, 4, 4])).shape
    [4, 4, 1, 1]

    """

    e = [jnp.newaxis] * (s.ndim - x.ndim)
    return x[tuple([...] + e)]


def register_car(cls):
    """
    Functionality wrapper to make custom car classes work.

    Notes
    -----
    `@register_car` is equivalent to `@dataclass` followed by
    `@jax.tree_util.register_pytree_node_class`.

    """

    cls = dataclass(cls)
    cls = jtu.register_pytree_node_class(cls)
    return cls


@dataclass
class Quantity:
    """
    Dataclass implementing physical quantities.

    Parameters
    ----------
    shape : List of ints
        the shape the array containing this quantity will take
        must not be an empty list

    name : String, optional
        the name of the quantity

    unit : String, optional
        the unit of the quantity

    """

    shape: List[int]
    unit: str = field(default=None)
    name: str = field(default=None)

    @property
    def axis_label(self):
        if self.unit is None:
            return self.name
        else:
            return f"{self.name} in $[{self.unit}]$"


@register_car
class BaseCar:
    """
    Base dataclass implementing how the car's physical quantities are
    represented. This car class should be subclassed when one is 
    aiming to use custom quantities.

    Examples
    --------

    @register_car
    class CustomCarClass(BaseCar):
        custom_quantity1: Quantity(shape1, unit1, name1)
        custom_quantity2: Quantity(shape2, unit2, name2)
        ...

    """

    t: Quantity([1], "s", "time")

    @classmethod
    def zeros(cls, batch_dims=[]):
        """
        A classmethod that returns a (batched) instance of the car
        class, where all quantity's values are set to zero.

        Parameters
        ----------
        batch_dims : List of ints, optional
            lenght of first axes independent of quantities' shapes

        Returns
        -------
        zeros_cls : instance of this car class
            quantities are of shape [`batch_dims`, `quantity.shape`]
            and set to zero everywhere.

        """

        children = [
            jnp.zeros(batch_dims + f.type.shape)
            for f in fields(cls)]

        return cls(*children)

    def tree_flatten(self):
        """ Utility function to use class as pytree """

        children = tuple(
            self.__dict__[f.name]
            for f in fields(self))

        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """ Utility function to use class as pytree """

        return cls(*children)

    @property
    def batch_dims(self):
        """
        Returns
        -------
        batch_dims : list of ints
            length of axes independent of quantities' shapes

        """

        value = self.__dict__[fields(self)[0].name]
        n = len(fields(self)[0].type.shape)
        return list(value.shape)[:-n]

    @property
    def quantities(self):
        """
        Returns
        -------
        quantities : tuple of quantities
            all quantities this class represents

        """
        return tuple(f.type for f in fields(self))

    @jax.jit
    def __add__(self, other):
        """
        Defines how two instances of this class can be added.

        Parameters
        ----------
        other: another instance of this car class
            to be added to this instance

        Returns
        -------
        added : an instance of this car class

        """
        return jtu.tree_map(
            lambda x, y: x + y, self, other)

    @jax.jit
    def __mul__(self, multiplier):
        """
        Defines how an this class can be multiplied by a scalar or
        an array, that is compatible with `batch_dims`.

        Parameters
        ----------
        multiplier: array_like
            this instance will be multiplied by this array

        Returns
        -------
        multiplied : an instance of this car class

        """
        return jtu.tree_map(
            lambda x: x * expand_like(multiplier, x), self)

    @property
    def shapes(self):
        """
        Returns
        -------
        shapes : dict of tuples of ints
            the shapes of the quantities' values
            should only be used for debugging purposes

        """
        return {
            f.name: self.__dict__[f.name].shape
            for f in fields(self)}


@register_car
class Car1D(BaseCar):
    x: Quantity([1], "m", "position")
    v: Quantity([1], "m/s", "velocity")
    g: Quantity([1], None, "gas")
