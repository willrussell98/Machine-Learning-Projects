###############################     NOTE     #####################################

# Need to install pygame on system and run on pycharm/vscode

# Save player and background pictures in folder for complete game experience

#############################     IMPORTS     ####################################

import pygame as pg
import random
import numpy as np
import matplotlib.pyplot as plt

#############################     SETTINGS     ####################################

# screen dimensions
TITLE = "Bee Game"

# full screen 1440 x 850
WIDTH = 1440
HEIGHT = 850
FPS = 500             # <- how fast the game runs

# define colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREY = ((50, 50, 50))
LIGHT_BLUE = ((200, 255, 255))
ORANGE = ((255, 127, 36))

# math vector initialisation for pygame
vec = pg.math.Vector2

# global variables
total_organisms = 500
global_moves = 500
checkpoint_x = 1175
checkpoint_y = 446
global_mutation_probability = 0.1
selection_pool = 250
number_of_generations = 10
threshold = 490


#############################    FUNCTIONS    #####################################


# function to check if the player collides with the checkpoint
def check_collision(player_x, player_y, player_width, player_height, checkpoint_x, checkpoint_y, checkpoint_width, checkpoint_height):
    if player_x > checkpoint_x and player_x < checkpoint_x + checkpoint_width or player_x + player_width > checkpoint_x and player_x + player_width < checkpoint_x + checkpoint_width:
        if player_y > checkpoint_y and player_y < checkpoint_y + checkpoint_height or player_y + player_height > checkpoint_y and player_y + player_height < checkpoint_y + checkpoint_height:
            return True


# create initial organisms
def all_objects():
    Make_Genes = []
    for i in range(total_organisms):
        object = np.random.multivariate_normal([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(
            -2, 2), random.uniform(-2, 2)], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], global_moves)

        Make_Genes.append(object)

    return Make_Genes


# define the fitness function
def FitnessFunction(X, Y):
    # Computes the Euclidean distance between X and Y
    euclidean_distance = np.linalg.norm(X - Y, axis=0)

    # to prevent a zero division error
    if euclidean_distance <= 35:
        euclidean_distance = 0.001

    # calculate fitness score
    fitness_score = 1 / euclidean_distance

    return fitness_score


# create a gene pool of the most successful objects
def GenePool(total_genes):

    Checkpoint = np.array([checkpoint_x, checkpoint_y])

    # list comprehension to create a fitness score for each object
    total_fitness_scores = [FitnessFunction(
        fitness[global_moves - 1][0:2], Checkpoint) for fitness in total_genes]
    # create dictionary for each chromsome and their fitness score
    gene_dictionary = {fitness: genes for genes,
                       fitness in zip(total_genes, total_fitness_scores)}
    # put the gene dictionary in descending order
    descending_gene_dictionary = dict(
        sorted(gene_dictionary.items(), key=lambda item: item[0], reverse=True))
    # take the top 250 highest values in the dictionary
    best_chromosomes = dict(
        list(descending_gene_dictionary.items())[:selection_pool])
    # first 250 chromosomes fitness scores
    gene_pool_fitness_scores = []
    # iterate through the keys and add them to a new list
    for i in best_chromosomes.keys():
        gene_pool_fitness_scores.append(i)
    # divide each fitness score by the total fitness score for roulette wheel selection probabilities an put it in an array
    fitness_scores_probabilities = np.array(
        [indvidual_fitness/sum(gene_pool_fitness_scores) for indvidual_fitness in gene_pool_fitness_scores])
    # create an array of the fitness scores of the first 10 chromosomes for random selection
    gene_fitness_scores = np.array(gene_pool_fitness_scores)

    return best_chromosomes, gene_fitness_scores, fitness_scores_probabilities


# global lists for later reference
gene_population = []
final_fitness_list = []
reached_checkpoint = []
highest_fitness_list = [0]
average_fitness_list = [0]
lowest_fitness_list = [0]
successful_organisms = [0]


#########################    GENETIC ALGORITHM    #################################

# create a class that represents the genetic algorithm
class GeneticAlgorithm():

    # initialise the class
    def __init__(self, chromosomes, gene_scores, fitness_scores_prob):
        self.chromosomes = chromosomes
        self.gene_scores = gene_scores
        self.fitness_scores_prob = fitness_scores_prob

    # roulette wheel selection function to select two parents
    def Roulette_wheel_selection(self):

        # empty list of the parents list
        parents = []

        # spin wheel twice to get two parents
        for spin_wheel in range(1, 3):
            # randomly choose a gene pool fitness score based upon it's probability of being selected
            Parent_Chosen = np.random.choice(
                self.gene_scores, p=self.fitness_scores_prob)
            parents.append(self.chromosomes[Parent_Chosen])

        return parents

    # crossover the chromosomes of the Parents
    def Crossover(self, Parent1, Parent2):

        # filter the chromosomes to remove the positions of the objects and only show the velocity and acceleration values
        Parent1 = Parent1[:, [2, 3, 4, 5]]
        Parent2 = Parent2[:, [2, 3, 4, 5]]

        # produce 2 children by splitting up the parents Chromosomes in quarters and concatenating the outputs
        Child1 = np.concatenate((np.array_split(Parent1, 4)[0], np.array_split(
            Parent2, 4)[1], np.array_split(Parent1, 4)[2], np.array_split(Parent2, 4)[3]))
        Child2 = np.concatenate((np.array_split(Parent2, 4)[0], np.array_split(
            Parent1, 4)[1], np.array_split(Parent2, 4)[2], np.array_split(Parent1, 4)[3]))

        # shuffle the genotype of each child to stop cyclical crossover issues
        np.random.shuffle(Child1)
        np.random.shuffle(Child2)

        return Child1, Child2

    # mutate a bit of the chromosome with a 10% probability
    def Mutation(self, Child1, Child2):

        DNA = [Child1, Child2]

        for chromosome in DNA:
            # the probability of mutation
            Mutation_Probability = random.uniform(0, 1)

            # does the chromosome mutate?
            if Mutation_Probability <= global_mutation_probability:

                # pick the bit that will be mutated between 1 and 10
                mutation_point = random.randrange(0, global_moves)

                # mutate the bit
                chromosome[mutation_point] = (
                    random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-2, 2), random.uniform(-2, 2))

            else:
                pass

        return DNA


# create new organisms for the next generation
def CreateNewObjects():

    # list that will contain the new objects
    new_objects = []

    for new_genes in range(1, total_organisms + 1):
        GA = GeneticAlgorithm(chromosomes=Successful_Objects,
                              gene_scores=Fitness_Scores, fitness_scores_prob=Fitness_Scores_Prob)

        parent1 = GA.Roulette_wheel_selection()[0]
        parent2 = GA.Roulette_wheel_selection()[1]

        child1 = GA.Crossover(parent1, parent2)[0]
        child2 = GA.Crossover(parent1, parent2)[1]

        New_Object_1 = GA.Mutation(child1, child2)[0]
        New_Object_2 = GA.Mutation(child1, child2)[1]

        # randomly choose a child from the potential children
        new_children = [New_Object_1, New_Object_2]
        chosen_child = random.sample(new_children, 1)

        new_objects.append(chosen_child[0])

    return new_objects


#############################     SPRITES     #####################################

# create player class that represents each bee
class Player(pg.sprite.Sprite):
    def __init__(self, genotype):
        pg.sprite.Sprite.__init__(self)
        self.genotype = genotype
        # self.image = pg.image.load('bee.png').convert_alpha()
        # self.image = pg.transform.scale(self.image, (20, 20))
        self.image = pg.Surface((10, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (100, HEIGHT / 2)
        self.pos = vec(160, 325)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.index = 0
        self.Column1 = self.genotype[:, 0]
        self.Column2 = self.genotype[:, 1]
        self.Column3 = self.genotype[:, 2]
        self.Column4 = self.genotype[:, 3]
        self.checkpoint = np.array([checkpoint_x, checkpoint_y])
        self.total_DNA = []

    # update for each move by the player
    def update(self):
        if self.index < global_moves:

            # if the player collides with the checkpoint
            if check_collision(self.rect.x, self.rect.y, 20, 20, checkpoint_x-30, checkpoint_y-30, 60, 60) == True:
                self.pos.x = self.rect.x
                self.pos.y = self.rect.y
                self.random_vel_x = 0
                self.random_vel_y = 0
                self.vel = vec(self.random_vel_x, self.random_vel_y)
                self.random_acc_x = 0
                self.random_acc_y = 0
                self.acc = vec(self.random_acc_x, self.random_acc_y)

                # if player collides with the checkpoint then increment the number of successful organisms
                if self.index == global_moves - 1:
                    reached_checkpoint.append(1)

            else:

                # move player across the screen
                self.random_vel_x = self.Column1[self.index]
                self.random_vel_y = self.Column2[self.index]
                self.vel = vec(self.random_vel_x, self.random_vel_y)
                self.random_acc_x = self.Column3[self.index]
                self.random_acc_y = self.Column4[self.index]
                self.acc = vec(self.random_acc_x, self.random_acc_y)

                # equations of motion

                self.pos += self.vel + 0.5 * self.acc

                # wrap around screen
                if self.pos.x > WIDTH:
                    self.pos.x = WIDTH - 10
                if self.pos.x < 0:
                    self.pos.x = 10
                if self.pos.y > 650:
                    self.pos.y = 650 - 10
                if self.pos.y < 0:
                    self.pos.y = 10

                # changing the position
                self.rect.center = self.pos

            # DNA information per update
            DNA = [self.pos[0], self.pos[1], self.vel[0],
                   self.vel[1], self.acc[0], self.acc[1]]

            # total DNA information of the player per generation
            self.total_DNA.append(DNA)

            # last move of the geneartion - find the fitness score and add it to the global list
            if self.index == global_moves - 1:
                gene_population.append(np.array(self.total_DNA))
                Position = np.array([self.pos[0], self.pos[1]])
                Fitness = FitnessFunction(Position, self.checkpoint)
                final_fitness_list.append(Fitness)

            # increment the update function
            self.index += 1


###############################     GAME     ######################################

# create game class
class Game():
    # initialise the game window
    def __init__(self):
        # initialise pygame
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        self.running = True
        self.scoreboard = pg.Rect(0, 650, 1440, 500)
        self.organism_moves = 0
        self.generation = 0
        self.highest_fitness = 0
        self.average_fitness = 0
        self.lowest_fitness = 0
        self.reached_checkpoint = 0
        #self.sunflower = pg.image.load("sunflower.png").convert_alpha()
        #self.sunflower = pg.transform.scale(self.sunflower, (70, 70))

    # start a new game

    def new(self, genes):

        self.generation += 1

        # create a sprite group, put every sprite into this group
        self.all_sprites = pg.sprite.Group()

        # create multiple players and spawn them into the game
        for index, objects in enumerate(genes):
            self.player = Player(genes[index])
            self.all_sprites.add(self.player)

        # run when a new game is started
        self.run()

    # run the game loop

    def run(self):
        self.playing = True
        while self.playing:
            # keep loop running at the correct speed
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    # game loop - update

    def update(self):
        self.all_sprites.update()

    # game loop - events

    def events(self):
        # process inputs (events)
        for event in pg.event.get():
            # check for closing the window
            if event.type == pg.QUIT:
                if self.playing:
                    self.playing = False
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    if self.playing:
                        self.playing = False
                    self.running = False

    # game loop - draw

    def draw(self):

        # Set the background color to black
        background_color = (0, 0, 0)  # Black color
        self.screen.fill(background_color)

        # Draw the checkpoint as a 20x20 square at the specified coordinates
        checkpoint_color = (255, 255, 0)  # Yellow color
        checkpoint_rect = pg.Rect(checkpoint_x - 10, checkpoint_y - 10, 20, 20)
        pg.draw.rect(self.screen, checkpoint_color, checkpoint_rect)

        # draw every player
        self.all_sprites.draw(self.screen)

        # draw the scoreboard
        pg.draw.rect(self.screen, GREY, self.scoreboard)

        # show scoreboard labels
        show(f"Generation: {self.generation}",
             600, 670, self.screen, colour=LIGHT_BLUE)
        show(f"Move: {self.organism_moves + 1}",
             1200, 670, self.screen, colour=LIGHT_BLUE)
        show(f"Successful Organisms: {self.reached_checkpoint}",
             10, 670, self.screen, colour=WHITE)
        show(f"Highest Fitness Score: {round(self.highest_fitness, 10)}",
             10, 720, self.screen, colour=GREEN)
        show(f"Average Fitness Score: {(round(self.average_fitness, 10))}",
             10, 770, self.screen, colour=ORANGE)
        show(f"Lowest Fitness Score: {round(self.lowest_fitness, 10)}",
             10, 820, self.screen, colour=RED)

        # after drawing everything, flip the display
        pg.display.flip()

        # increment the number of moves
        self.organism_moves += 1

        # auto restart the game at the end of every generation
        if self.organism_moves == global_moves:
            self.organism_moves = 0
            self.highest_fitness = max(final_fitness_list)
            highest_fitness_list.append(self.highest_fitness)
            self.average_fitness = np.mean(final_fitness_list)
            average_fitness_list.append(self.average_fitness)
            self.lowest_fitness = min(final_fitness_list)
            lowest_fitness_list.append(self.lowest_fitness)
            self.reached_checkpoint = len(reached_checkpoint)
            successful_organisms.append(self.reached_checkpoint)
            self.playing = False
            self.running = False


# initialise an instance of the game
game = Game()

# define font function for pygame
font = pg.font.Font(None, 30)

# upload images for the background
# background_image = pg.image.load("trees.jpeg").convert()
# background_image = pg.transform.scale(background_image, (1440, 650))
# background_rect = background_image.get_rect()


# function that displays text
def show(text, x, y, screen, colour=WHITE):
    screen.blit(font.render(text, True, colour), (x, y))


# run the game
while game.running:

    # create initial organisms and label these as the population
    population = all_objects()

    while game.reached_checkpoint < threshold:

        # break the loop if the maximum number of generations is met
        if game.generation == number_of_generations:
            break

        else:
            # initialise a new generation using the population
            game.new(population)

            # extract information from the gene pool
            Genotype = GenePool(gene_population)

            # genotype - gene dictionary
            Successful_Objects = Genotype[0]

            # genotype - gene fitness scores
            Fitness_Scores = Genotype[1]

            # genotype - gene weighted fitness scores
            Fitness_Scores_Prob = Genotype[2]

            # create a new population
            population = CreateNewObjects()

            # reset global lists
            reached_checkpoint.clear()

    # four subplots to display the summary statistics of the Genetic Algorithm
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # average fitness score plot
    axs[0, 0].plot(average_fitness_list, color='orange')
    axs[0, 0].set_title('Average Fitness Scores')
    axs[0, 0].set(xlabel='Generation', ylabel='Fitness Score')

    # highest fitness score plot
    axs[0, 1].plot(highest_fitness_list, color='green')
    axs[0, 1].set_title('Highest Fitness Scores')
    axs[0, 1].set(xlabel='Generation', ylabel='Fitness Score')

    # lowest fitness score plot
    axs[1, 0].plot(lowest_fitness_list, color='green')
    axs[1, 0].set_title('Lowest Fitness Scores')
    axs[1, 0].set(xlabel='Generation', ylabel='Fitness Score')

    # successful organism score plot
    axs[1, 1].plot(successful_organisms, color='black')
    axs[1, 1].set_title('Total Succeeding Organisms')
    axs[1, 1].set(xlabel='Generation',
                  ylabel='Successful Organisms')

    fig.tight_layout(pad=3.0)

    # show graph
    plt.show()

    gene_population.clear()
    final_fitness_list.clear()

pg.quit()
