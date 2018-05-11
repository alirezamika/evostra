Evostra: Evolution Strategy for Python
--------

Evolution Strategy (ES) is an optimization technique based on ideas of adaptation and evolution.
You can learn more about it at https://blog.openai.com/evolution-strategies/

Installation
--------
It's compatible with both python2 and python3.

Install from source:

.. code-block:: bash

    $ python setup.py install

    
Install latest version from git repository using pip:

.. code-block:: bash

    $ pip install git+https://github.com/alirezamika/evostra.git
    
    
Install from PyPI:

.. code-block:: bash

    $ pip install evostra
    
(You may need to use python3 or pip3 for python3)


Sample Usages
--------

`An AI agent learning to play flappy bird using evostra 
<https://github.com/alirezamika/flappybird-es>`_


`An AI agent learning to walk using evostra 
<https://github.com/alirezamika/bipedal-es>`_


How to use
--------

The input weights of the EvolutionStrategy module is a list of arrays (one array with any shape for each layer of the neural network), so we can use any framework to build the model and just pass the weights to ES.


For example we can use Keras to build the model and pass its weights to ES, but here we use Evostra's built-in model FeedForwardNetwork which is much faster for our use case:


.. code:: python

    import numpy as np
    from evostra import EvolutionStrategy
    from evostra.models import FeedForwardNetwork

    # A feed forward neural network with input size of 5, two hidden layers of size 4 and output of size 3
    model = FeedForwardNetwork(layer_sizes=[5, 4, 4, 3])


Now we define our get_reward function:

.. code:: python

    solution = np.array([0.1, -0.4, 0.5])
    inp = np.asarray([1, 2, 3, 4, 5])

    def get_reward(weights):
        global solution, model, inp
        model.set_weights(weights)
        prediction = model.predict(inp)
        # here our best reward is zero
        reward = -np.sum(np.square(solution - prediction))
        return reward


Now we can build the EvolutionStrategy object and run it for some iterations:

.. code:: python

    # if your task is computationally expensive, you can use num_threads > 1 to use multiple processes;
    # if you set num_threads=-1, it will use number of cores available on the machine; Here we use 1 process as the
    #  task is not computationally expensive and using more processes would decrease the performance due to the IPC overhead.
    es = EvolutionStrategy(model.get_weights(), get_reward, population_size=20, sigma=0.1, learning_rate=0.03, decay=0.995, num_threads=1)
    es.run(1000, print_step=100)


Here's the output:

.. code::

    iter 100. reward: -68.819312
    iter 200. reward: -0.218466
    iter 300. reward: -0.110204
    iter 400. reward: -0.001901
    iter 500. reward: -0.000459
    iter 600. reward: -0.000287
    iter 700. reward: -0.000939
    iter 800. reward: -0.000504
    iter 900. reward: -0.000522
    iter 1000. reward: -0.000178
    
    
Now we have the optimized weights and we can update our model:

.. code:: python
    
    optimized_weights = es.get_weights()
    model.set_weights(optimized_weights)
    

Todo
--------
- Add distribution support over network
