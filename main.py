from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psns

from reconstructor import Reconstructor
from draw import draw_square, draw_line, draw_rectangle
from crossover import (
    random_vertical_swap, 
    random_horizontal_swap, 
    half_vertical_swap, 
    half_horizontal_swap, 
    blend
)


class MyReconstructor(Reconstructor):
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
        super().__init__(
            image, 
            n_generations, 
            n_population, 
            mutation_chance,
            mutation_strength,
            start_mutation_strength,
            use_elitism,
            n_elitism,
            print_every,
            gif_save_every,
        )
    
    def _evaluate_fitness(self, image: np.ndarray) -> float:
        return psns(self.original_image, image)

    def _add_random_elements(self, image: np.ndarray, k: int) -> np.ndarray:
        for _ in range(k):
            image = draw_square(image)
        return image

    def _crossover(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        return random_horizontal_swap(image1, image2)


def main(args: Namespace):
    reconstructor = MyReconstructor(
        image=args.image,
        n_generations=args.n_gen,
        n_population=args.n_pop,
        mutation_chance=args.mut_chance,
        mutation_strength=args.mut_strength,
        start_mutation_strength=args.start_mut_strength,
        use_elitism=not args.wo_elitism,
        n_elitism=args.n_elitism,
    )
    reconstructor.run()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', type=Path)
    parser.add_argument('--n-gen', type=int, default=3000)
    parser.add_argument('--n-pop', type=int, default=100)
    parser.add_argument('--mut-chance', type=float, default=0.1),
    parser.add_argument('--mut-strength', type=int, default=3),
    parser.add_argument('--start-mut-strength', type=int, default=6)
    parser.add_argument('--wo-elitism', action='store_true')
    parser.add_argument('--n-elitism', type=int, default=20)

    command_line_args = parser.parse_args()
    main(command_line_args)
