Evostra: Evolution Strategy for Python
--------

Evolutio Strategy (ES) is an optimization technique based on ideas of adaptation and evolution.
You can learn more about it at https://blog.openai.com/evolution-strategies/

Installation
--------
It's compatible with both python2 and python3.

Install from source:

.. code-block:: bash

    $ sudo python setup.py install

    
Install from PyPI:

.. code-block:: bash

    $ sudo pip install evostra
    
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


Here we use Keras to build the model and we pass its weights to ES.

.. code:: python

    from evostra import EvolutionStrategy
    from keras.models import Model, Input
    from keras.layers import Dense
    from keras.optimizers import Adam # not important as there's no training here.
    import numpy as np
    
    input_layer = Input(shape=(5,1))
    layer = Dense(8)(input_layer)
    output_layer = Dense(3)(layer)
    model = Model(input_layer, output_layer)
    model.compile(Adam(), 'mse')
  
  
  
  
Now we define our get_reward function:

.. code:: python
    
    solution = np.array([0.1, -0.4, 0.5])
    inp = np.asarray([[1,2,3,4,5]])
    inp = np.expand_dims(inp, -1)
   
    def get_reward(weights):
        global solution, model, inp
        model.set_weights(weights)
        prediction = model.predict(inp)[0]
        # here our best reward is zero
        reward = -np.sum(np.square(solution - prediction))
        return reward
    
    
Now we can build the EvolutionStrategy object and run it for some iterations:

.. code:: python

    es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.001, decay=0.993)
    es.run(1000, print_step=100)
    
    
Here's the output:

.. code::

    iter 0. reward: -68.819312
    iter 100. reward: -0.218466
    iter 200. reward: -0.110204
    iter 300. reward: -0.089003
    iter 400. reward: -0.078224
    iter 500. reward: -0.063891
    iter 600. reward: -0.049090
    iter 700. reward: -0.027701
    iter 800. reward: -0.013094
    iter 900. reward: -0.009140
    
    
Now we have the optimized weights and we can update our model:

.. code:: python
    
    optimized_weights = es.get_weights()
    model.set_weights(optimized_weights)
    

Todo
--------
- Add distribution (multi-cpu) support
