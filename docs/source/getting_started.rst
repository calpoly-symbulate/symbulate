Getting Started
===============

Install Symbulate with:

.. code-block:: console

   pip install symbulate

Create a probability model, define a random variable, and simulate it:

.. code-block:: python

   from symbulate import *

   X = RV(Bernoulli(0.5))
   results = X.sim(100)
   print(results.tabulate())

See the :doc:`api/modules` for the complete API reference.
