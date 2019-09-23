import pygame
import random
import time
import numpy as np

import matplotlib.pyplot as plt
from genome import Genome
from network import Network
from population import Population
from config import Config

global fitnessovergeneration
fitnessovergeneration = []
global fittestovergeneration
fittestovergeneration = []
global fittest

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500
speed = 5

def playGame(genome):
	playerX = random.randint(5, SCREEN_WIDTH)
	playerY = random.randint(5, SCREEN_HEIGHT)

	DestX = random.randint(20, SCREEN_WIDTH)
	DestY = random.randint(20, SCREEN_HEIGHT)

	seconds=0
	while True:
		seconds += 1
		if seconds > 1000:
			break
		else:
			DiffX = playerX - DestX
			DiffY = playerY - DestY

			NNinput = np.array([[DiffX],[DiffY]])

			NNoutput = genome.network.feedforward([[DiffX], [DiffY]])

			hori = NNoutput[0][0]
			vert = NNoutput[1][0]

			if hori > 0.5:
				playerX -= speed
			else:
				playerX += speed
			if vert > 0.5:
				playerY -= speed
			else:
				playerY += speed

			if abs(playerX - DestX) < 5 and abs(playerY - DestY) < 5:
				genome.fitness += 1

	return genome

global bestFitness
bestFitness = 0
population = Population()
population.generateRandomPopulation()
generation = 1
maxgeneration = Config.maxGeneration
lastgenerationaveragefitness = 0

while generation <= maxgeneration:

	for i in range(population.size()):

		genome = playGame(population.getGenome(i))
		population.setGenomeFitness(i,genome.fitness)

		if genome.fitness > bestFitness:
		    bestFitness = genome.fitness
		    fittest = genome

		fitnessovergeneration.append(population.averageFitness())

		lastgenerationaveragefitness = population.averageFitness()

		fittestovergeneration.append(population.findFittest().fitness)
		#Evolve the population
		population.evolvePopulation()
		generation += 1
	print(f'Generation: {generation}, Fittest: {bestFitness}')

print('Diplaying Fittest Individual...')

pygame.init()
win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
run = True

playerX = random.randint(5, SCREEN_WIDTH)
playerY = random.randint(5, SCREEN_HEIGHT)

DestX = random.randint(20, SCREEN_WIDTH)
DestY = random.randint(20, SCREEN_HEIGHT)

while run:
	pygame.time.delay(50)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	DiffX = playerX - DestX
	DiffY = playerY - DestY

	NNinput = np.array([[DiffX],[DiffY]])
	NNoutput = fittest.network.feedforward([[DiffX], [DiffY]])

	hori = NNoutput[0][0]
	vert = NNoutput[1][0]

	if hori > 0.5:
		playerX -= speed
	else:
		playerX += speed
	if vert > 0.5:
		playerY -= speed
	else:
		playerY += speed

	if abs(playerX - DestX) < 10 and abs(playerY - DestY) < 10:
		print('Catch')
		print('Randomizing...')
		playerX = random.randint(5, SCREEN_WIDTH)
		playerY = random.randint(5, SCREEN_HEIGHT)

		DestX = random.randint(20, SCREEN_WIDTH)
		DestY = random.randint(20, SCREEN_HEIGHT)
		win.fill((0,0,0))
	if playerX < 0 or playerX > SCREEN_WIDTH or playerY < 0 or playerY > SCREEN_HEIGHT:
		print('Reached Out of Bounds. Randomizing...')
		playerX = random.randint(5, SCREEN_WIDTH)
		playerY = random.randint(5, SCREEN_HEIGHT)

	pygame.draw.rect(win, (255, 25, 25), pygame.Rect(playerX, playerY, 10, 10))
	pygame.draw.rect(win, (116, 255, 91), pygame.Rect(DestX, DestY, 10,10))
	pygame.display.flip()

