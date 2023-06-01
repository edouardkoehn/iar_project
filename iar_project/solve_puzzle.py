## load images
import os
from time import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from solving_puzzle.genetic_algorithm import GeneticAlgorithm

folder = "puzzle_unsolve"
filename = "test.jpg"


def create_puzzle(image):
    if image.shape[0] > image.shape[1]:
        rows = 4
        cols = 3
    elif image.shape[1] == 512 and image.shape[0] == 512:
        rows = 4
        cols = 4
    else:
        rows = 3
        cols = 3

    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Calculate the dimensions of each puzzle image
    puzzle_height = height // rows
    puzzle_width = width // cols

    # Create an empty list to store the puzzle images
    puzzle_pieces = []

    # Iterate through each row and column of the image to extract puzzle images
    for i in range(rows):
        for j in range(cols):
            # Calculate the starting and ending coordinates for each puzzle image
            start_row = i * puzzle_height
            end_row = start_row + puzzle_height
            start_col = j * puzzle_width
            end_col = start_col + puzzle_width

            # Extract the puzzle image from the original image
            puzzle_image = image[start_row:end_row, start_col:end_col, :]

            # Convert the numpy array back to a PIL Image object
            puzzle_image = Image.fromarray(puzzle_image)

            # Append the puzzle image to the list
            puzzle_pieces.append(puzzle_image)

    return puzzle_pieces


def solve_puzzle(
    image_unsolve,
    piece_size=128,
    population=600,
    generations=20,
    termination_threshold=10,
    verbose=False,
    save=False,
):

    # print("\n=== Population:  {}".format(population))
    # print("=== Generations: {}".format(generations))
    # print("=== Piece size:  {} px".format(piece_size))

    # Let the games begin! And may the odds be in your favor!
    start = time()
    algorithm = GeneticAlgorithm(
        image_unsolve, piece_size, population, generations, termination_threshold
    )
    solution = algorithm.start_evolution(verbose)
    end = time()

    # print("\n=== Done in {0:.3f} s".format(end - start))

    solution_image = solution.to_image().astype(np.uint8)
    return solution_image
