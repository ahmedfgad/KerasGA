# KerasGA: Training Keras Model using the Genetic Algorithm
[KerasGA](https://github.com/ahmedfgad/KerasGA) is part of the [PyGAD](https://pypi.org/project/pygad) library for training Keras models using the genetic algorithm (GA). 

The [KerasGA](https://github.com/ahmedfgad/KerasGA) project has a single module named `kerasga.py` which has a class named `KerasGA` for preparing an initial population of Keras model parameters.

[PyGAD](https://pypi.org/project/pygad) is an open-source Python library for building the genetic algorithm and training machine learning algorithms. Check the library's documentation at [Read The Docs](https://pygad.readthedocs.io/): https://pygad.readthedocs.io

Before using this project, install [PyGAD](https://pypi.org/project/pygad) via pip:

```python
pip install pygad
```

# Donation

You can donate via [Open Collective](https://opencollective.com/pygad): [opencollective.com/pygad](https://opencollective.com/pygad). 

To donate using PayPal, use either this link: [paypal.me/ahmedfgad](https://paypal.me/ahmedfgad) or the e-mail address ahmed.f.gad@gmail.com.

# Installation

To install [PyGAD](https://pypi.org/project/pygad), simply use pip to download and install the library from [PyPI](https://pypi.org/project/pygad) (Python Package Index). The library lives a PyPI at this page https://pypi.org/project/pygad.

For Windows, issue the following command:

```python
pip install pygad
```

For Linux and Mac, replace `pip` by use `pip3` because the library only supports Python 3.

```python
pip3 install pygad
```

PyGAD is developed in Python 3.7.3 and depends on NumPy for creating and manipulating arrays and Matplotlib for creating figures. The exact NumPy version used in developing PyGAD is 1.16.4. For Matplotlib, the version is 3.1.0.

To get started with PyGAD, please read the documentation at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Source Code

The source code of the `PyGAD` modules is found in the following GitHub projects:

- [pygad](https://github.com/ahmedfgad/GeneticAlgorithmPython): (https://github.com/ahmedfgad/GeneticAlgorithmPython)
- [pygad.nn](https://github.com/ahmedfgad/NumPyANN): https://github.com/ahmedfgad/NumPyANN
- [pygad.gann](https://github.com/ahmedfgad/NeuralGenetic): https://github.com/ahmedfgad/NeuralGenetic
- [pygad.cnn](https://github.com/ahmedfgad/NumPyCNN): https://github.com/ahmedfgad/NumPyCNN
- [pygad.gacnn](https://github.com/ahmedfgad/CNNGenetic): https://github.com/ahmedfgad/CNNGenetic
- [pygad.kerasga](https://github.com/ahmedfgad/KerasGA): https://github.com/ahmedfgad/KerasGA

The documentation of PyGAD is available at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Documentation

The documentation of the PyGAD library is available at [Read The Docs](https://pygad.readthedocs.io) at this link: https://pygad.readthedocs.io. It discusses the modules supported by PyGAD, all its classes, methods, attribute, and functions. For each module, a number of examples are given.

If there is an issue using PyGAD, feel free to post at issue in this [GitHub repository](https://github.com/ahmedfgad/GeneticAlgorithmPython) https://github.com/ahmedfgad/GeneticAlgorithmPython or by sending an e-mail to ahmed.f.gad@gmail.com. 

If you built a project that uses PyGAD, then please drop an e-mail to ahmed.f.gad@gmail.com with the following information so that your project is included in the documentation.

- Project title
- Brief description
- Preferably, a link that directs the readers to your project

Please check the **Contact Us** section for more contact details.

# Life Cycle of PyGAD

The next figure lists the different stages in the lifecycle of an instance of the `pygad.GA` class. Note that PyGAD stops when either all generations are completed or when the function passed to the `on_generation` parameter returns the string `stop`.

![PyGAD Lifecycle](https://user-images.githubusercontent.com/16560492/89446279-9c6f8380-d754-11ea-83fd-a60ea2f53b85.jpg)

The next code implements all the callback functions to trace the execution of the genetic algorithm. Each callback function prints its name.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

fitness_function = fitness_func

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop)

ga_instance.run()
```

Based on the used 3 generations as assigned to the `num_generations` argument, here is the output.

```
on_start()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_stop()
```

# Examples

Check the [PyGAD's documentation](https://pygad.readthedocs.io/en/latest/README_pygad_gacnn_ReadTheDocs.html) for more examples information. You can also find more information about the implementation of the examples.

## Example 1: Train Keras Regression Model

```python
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)
    
    predictions = model.predict(data_inputs)
    mae = tensorflow.keras.losses.MeanAbsoluteError()
    abs_error = mae(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0/abs_error

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

input_layer  = tensorflow.keras.layers.Input(3)
dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

# Data inputs
data_inputs = numpy.array([[0.02, 0.1, 0.15],
                           [0.7, 0.6, 0.8],
                           [1.5, 1.2, 1.7],
                           [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = numpy.array([[0.1],
                            [0.6],
                            [1.3],
                            [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error)
```

## Example 2: XOR Binary Classification

```python
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)

    model.set_weights(weights=model_weights_matrix)

    predictions = model.predict(data_inputs)

    bce = tensorflow.keras.losses.BinaryCrossentropy()
    solution_fitness = 1.0 / (bce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Build the keras model using the functional API.
input_layer  = tensorflow.keras.layers.Input(2)
dense_layer = tensorflow.keras.layers.Dense(4, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(2, activation="softmax")(dense_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

# XOR problem inputs
data_inputs = numpy.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

# XOR problem outputs
data_outputs = numpy.array([[1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights.
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                              weights_vector=solution)
model.set_weights(best_solution_weights)
predictions = model.predict(data_inputs)
print("Predictions : \n", predictions)

# Calculate the binary crossentropy for the trained model.
bce = tensorflow.keras.losses.BinaryCrossentropy()
print("Binary Crossentropy : ", bce(data_outputs, predictions).numpy())

# Calculate the classification accuracy for the trained model.
ba = tensorflow.keras.metrics.BinaryAccuracy()
ba.update_state(data_outputs, predictions)
accuracy = ba.result().numpy()
print("Accuracy : ", accuracy)
```

# For More Information

There are different resources that can be used to get started with the building CNN and its Python implementation. 

## Tutorial: Implementing Genetic Algorithm in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Genetic Algorithm Implementation in Python**](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6)
- [KDnuggets](https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html)

[This tutorial](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) is prepared based on a previous version of the project but it still a good resource to start with coding the genetic algorithm.

[![Genetic Algorithm Implementation in Python](https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png)](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)

## Tutorial: Introduction to Genetic Algorithm

Get started with the genetic algorithm by reading the tutorial titled [**Introduction to Optimization with Genetic Algorithm**](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)
* [Towards Data Science](https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html)
* [KDnuggets](https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b)

[![Introduction to Genetic Algorithm](https://user-images.githubusercontent.com/16560492/82078259-26252d00-96e1-11ea-9a02-52a99e1054b9.jpg)](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)

## Tutorial: Build Neural Networks in Python

Read about building neural networks in Python through the tutorial titled [**Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset**](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad) available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)
* [Towards Data Science](https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491)
* [KDnuggets](https://www.kdnuggets.com/2019/02/artificial-neural-network-implementation-using-numpy-and-image-classification.html)

[![Building Neural Networks Python](https://user-images.githubusercontent.com/16560492/82078281-30472b80-96e1-11ea-8017-6a1f4383d602.jpg)](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)

## Tutorial: Optimize Neural Networks with Genetic Algorithm

Read about training neural networks using the genetic algorithm through the tutorial titled [**Artificial Neural Networks Optimization using Genetic Algorithm with Python**](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e)
- [KDnuggets](https://www.kdnuggets.com/2019/03/artificial-neural-networks-optimization-genetic-algorithm-python.html)

[![Training Neural Networks using Genetic Algorithm Python](https://user-images.githubusercontent.com/16560492/82078300-376e3980-96e1-11ea-821c-aa6b8ceb44d4.jpg)](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)

## Tutorial: Building CNN in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Building Convolutional Neural Network using NumPy from Scratch**](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a)
- [KDnuggets](https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html)
- [Chinese Translation](http://m.aliyun.com/yunqi/articles/585741)

[This tutorial](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)) is prepared based on a previous version of the project but it still a good resource to start with coding CNNs.

[![Building CNN in Python](https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png)](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)

## Tutorial: Derivation of CNN from FCNN

Get started with the genetic algorithm by reading the tutorial titled [**Derivation of Convolutional Neural Network from Fully Connected Network Step-By-Step**](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)
* [Towards Data Science](https://towardsdatascience.com/derivation-of-convolutional-neural-network-from-fully-connected-network-step-by-step-b42ebafa5275)
* [KDnuggets](https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html)

[![Derivation of CNN from FCNN](https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png)](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)

## Book: Practical Computer Vision Applications Using Deep Learning with CNNs

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

Find the book at these links:

- [Amazon](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)
- [Springer](https://link.springer.com/book/10.1007/978-1-4842-4167-7)
- [Apress](https://www.apress.com/gp/book/9781484241660)
- [O'Reilly](https://www.oreilly.com/library/view/practical-computer-vision/9781484241677)
- [Google Books](https://books.google.com.eg/books?id=xLd9DwAAQBAJ)

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

# Contact Us

* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Amazon Author Page](https://amazon.com/author/ahmedgad)
* [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
