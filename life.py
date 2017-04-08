import numpy as np
from itertools import combinations
from genes import gene_pool
from random import choice, shuffle
from functools import partial

GENOME_SIZE = 15
POPULATION_SIZE = 20
N_TRIALS = 5

def map_funcs(obj, func_list):
    return [func(obj) for func in func_list][0]


def sample_lucky(fit_vals):
    '''Given fittness values sample from multinomial'''
    fit_vals = np.array(fit_vals).astype('float')
    normalized = fit_vals/sum(fit_vals)
    parents = np.random.multinomial(N_TRIALS, normalized)
    return np.where(parents != 0)[0]


def generate_couples(fit_vals):
    '''Given fitness values returns parent indices.'''
    parents = []
    while len(parents) < POPULATION_SIZE/2:
        parents += list(combinations(sample_lucky(fit_vals), 2))
    shuffle(parents)
    return parents[:POPULATION_SIZE/2]



def first_inhabitants():
    '''Generate seed population.'''
    population = []
    for i in range(POPULATION_SIZE):
        genome = []
        for j in range(GENOME_SIZE):
            op = choice(gene_pool)
            if op.func_name.startswith('adjust'):
                low, high = np.random.random(2)
                size = np.random.randint(1, 20)
                gene = partial(op, values=np.random.uniform(low, high,
                                                            size=size))
            elif op.func_name == 'sharpen':
                a, b = np.random.random(2)
                sigma = np.random.uniform(0, 3.0)
                gene = partial(op, a=a, b=b, sigma=sigma)
            else:
                gene = op
            genome.append(gene)
        population.append(genome)
    return population


def ox1(p1, p2):
    '''
    Order Crossover Operator.

    From http://ictactjournals.in/paper/IJSC_V6_I1_paper_4_pp_1083_1092.pdf
    '''
    i, j = 0, 0
    while abs(i - j) < POPULATION_SIZE/4:
        i, j = np.random.randint(1, GENOME_SIZE, 2)
    if i > j:
         i, j = j, i
    c1 = [None for x in range(GENOME_SIZE)]
    c2 = [None for x in range(GENOME_SIZE)]
    c1[i:j] = p1[i:j]
    c2[i:j] = p2[i:j]
    c1[j:] = p2[:GENOME_SIZE-j]
    c1[0:i] = p2[GENOME_SIZE-j:i]
    c2[j:] = p1[:GENOME_SIZE-j]
    c2[0:i] = p1[GENOME_SIZE-j:i]
    return c1, c2


def evaluate_population():
    '''Pick a sample from the population and query the user for evaluation.'''
    not_visited = np.where(visited == 0)[0]
    print len(not_visited)
    if len(not_visited) == 0:
        print 'Done'
        return
    selection = np.random.choice(not_visited,
                                 min(len(not_visited), 3),
                                 replace=False)
    original_image = skimage.img_as_float(skimage.io.imread("plum2.jpg"))
    for i in selection:
        individual = population[i]
        img = map_funcs(original_image, individual)
        plt.imshow(img)
        plt.axis('off')

        plt.show()
        fit = raw_input("How much do you like the image from 1 to 5")
        print i
        fitness[i] = int(fit)
        visited[i] = 1
        display.clear_output(wait=True)
        display.display(plt.gcf())

def selection():
    '''Replace 2 worst individuals with kids from the 2 best.'''
    evaluated = np.where(visited)[0]
    l = np.argsort(fitness[evaluated])[::-1]
    p1 = population[evaluated[l[0]]]
    p2 = population[evaluated[l[1]]]
    c1, c2 = ox1(p1, p2)
    population[l[-1]] = c1
    population[l[-2]] = c2
    fitness[l[-1]] = 0
    fitness[l[-2]] = 0
    visited[l[-1]] = 0
    visited[l[-2]] = 0

if __name__ == '__main__';
    population = first_inhabitants()
    visited = np.zeros(POPULATION_SIZE)
    fitness = np.zeros(POPULATION_SIZE)
    for i in range(10):
        evaluate_population()
        selection()
        print fitness
        print visited
