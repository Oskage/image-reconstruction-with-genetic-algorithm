from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy.special import softmax
from tqdm import tqdm


class Reconstructor(ABC):
    def __init__(
        self, 
        image: Path, 
        n_generations: int, 
        n_population: int, 
        mutation_chance: float,
        mutation_strength: int,
        start_mutation_strength: int,
        use_elitism: bool,
        n_elitism: int,
        print_every: int = 25,
        gif_save_every: int = 25,
    ):
        self.original_image = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2GRAY)
        self.n_generations = n_generations
        self.n_population = n_population
        self.start_mutation_strength = start_mutation_strength
        self.mutation_strength = mutation_strength
        self.mutation_chance = mutation_chance
        self.use_elitism = use_elitism
        self.n_elitism = n_elitism
        self.print_every = print_every
        self.gif_save_every = gif_save_every
    
    def run(self):
        gif_frames = []

        population = self._create_first_population()

        for gen_idx in tqdm(range(self.n_generations), total=self.n_generations):
            fitnesses = np.array([self._evaluate_fitness(img) for img in population])

            new_population = np.zeros_like(population)
            
            top_population_indices = np.argsort(fitnesses)[-self.n_elitism:]
            if self.use_elitism:
                for idx, top_idx in enumerate(top_population_indices):
                    new_population[idx] = population[top_idx]
            
            parents_pairs = self._generate_parents_pairs(population, fitnesses)

            for individual_idx in range(self.n_population - self.n_elitism):
                new_img = self._crossover(*parents_pairs[individual_idx])
                
                if np.random.uniform() < self.mutation_chance:
                    new_img = self._add_random_elements(new_img, self.mutation_strength)
                
                new_population[self.n_elitism + individual_idx] = new_img
            
            if gen_idx % self.print_every == 0:
                top_fitness = fitnesses[top_population_indices[0]]
                print(f'gen#{gen_idx} best_fitness={top_fitness}')

            if gen_idx % self.gif_save_every == 0:
                gif_frames.append(population[top_population_indices[0]])
            
            population = new_population
        
        imageio.mimsave('out_gif.gif', gif_frames)
        cv2.imwrite('out_best.jpg', population[top_population_indices[-1]])

    def _create_first_population(self) -> np.ndarray:
        population = np.zeros((self.n_population, *self.original_image.shape), 
                              dtype=np.uint8)
        
        for idx in range(self.n_population):
            population[idx] = self._add_random_elements(population[idx], 
                                                        self.mutation_strength)
        
        return population

    def _generate_parents_pairs(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
    ) -> np.ndarray:
        individuals_number = fitnesses.shape[0]
        parents = np.zeros(shape=(individuals_number, 2, *population[0].shape), 
                           dtype=np.uint8)
        softmax_fitnesses = softmax(fitnesses)

        for idx, _ in enumerate(population):
            random_pair_indices = np.random.choice(individuals_number, size=2, 
                                                   p=softmax_fitnesses, replace=False)
            parents[idx] = population[random_pair_indices]
    
        return parents

    @abstractmethod
    def _evaluate_fitness(self, image: np.ndarray) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _add_random_elements(self, image: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _crossover(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
