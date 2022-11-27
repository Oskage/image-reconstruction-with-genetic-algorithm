# Image Reconstruction with Genetic Algorithm

## Description

Simple implementation of genetic algorithm image reconstruction

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py <path_to_image>
```

## Detailed info

Reconstructor is an abstract class. You need to implement an child of Reconstructor, which
forces you to implement few functions. _evaluate_fitness takes two image and produces 
a score of fitness or similarity. _add_random_elements is a mutation function, in which
you need to take an image and do something, that changes it a little. And _crossover that
takes two images and mixes those to get their "child".You are free to modify algorithm 
implementation in Reconstructor but be careful.
