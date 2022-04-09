### ALl packages import
import numpy as np
import random
import matplotlib.pyplot as plt
from autoencoder import *


#All function definition

def generation_copies(X,n):
    '''This function generates n copies of the parent X (that will serve to create the final children)
    Args:
        X (np.array) = vector of the latent space of a parent
        n (int) = number of copies we want from this parent
    Return :
        A np.array containing n copies of X '''
    X2 = list(X)
    Y = []
    for i in range(n):
        Y.append(X2)
    return np.array(Y)



def mutation_un_vecteur(parent, step_size, p):
    ''' This function applies a mutation on each element of the latent_vector in each subnp.array of the list Y containing copies of the same parent or recombinations of parents
    Args:
        parent (np.array) : a vector of the latent space
        step_size (float) : parameter of the mutation
        p (float) : probability (between 0 and 1) that there is a mutation for each element of the latent vector
    Return :
        x (np.array) : vecteur modifié muté (avec une probabilité p de mutation pour chaque élément)'''
    n = len(parent)
    for i in range(n):
        if np.random.rand() < p:
            rand = np.random.randn()
            parent[i] += rand*step_size
    return parent



def mutation(Y, step_size, p):
    ''' This function applies a mutation on each vector contained in Y
    Args:
        Y (np.array) : np.array of vectors (latent vectors)
        step_size (float) : parameter of the mutation
        p (float) : probability (between 0 and 1) that there is a mutation for each element of the latent vector
    Return :
        Y (np.array) : the modification is made directly on Y, and at the end, Y contains len(Y) latent vectors, modified by mutation'''
    n = len(Y)
    for i in range(n):
        Y[i] = mutation_un_vecteur(Y[i], step_size, p)
    return Y



def recombination_2parents_moy(X1, X2, p):
    '''Calcule le vecteur obtenu à partir d'une recombinaison à partir des 2 parents X1 et X2, qui calcule la moyenne des coordonnées pour chaque élément des vecteurs latents, avec une probabilité p pour chaque élément
        Sinon, c'est l'un ou l'autre des éléments de X1 ou X2 qui est choisi
    Args :
        X1 (np.array/list): vecteur de l'espace latent, un parent
        X2 (np.array/list) : vecteur de l'espace latent, un parent
        p (float) : probabilité de modifier l'élément à chaque tour de boucle (p entre 0 et 1, si p vaut 1, new_vector est la moyenne de X1 et X2, si p vaut 0, les éléments de new_vector sont extraits soit de X1, soit de X2)
    Return :
        new_vector(np.array) : np.array formant un nouveau vecteur de l'espace latent, avec pour certaines coordonnées celles de X1, pour d'autres celles de X2, pour d'autres la moyenne des 2'''
    n = len(X1)
    new_vector = []
    for i in range(n):
        if np.random.rand() < p :
            x = (X1[i]+X2[i])/2
            new_vector.append(x)
        else :
            if np.random.rand() < 1/2 :
                new_vector.append(X1[i])
            else :
                new_vector.append(X2[i])
    return np.array(new_vector)



def recombination_k_parents(X, p):
    '''Calcule le vecteur obtenu à partir d'une recombinaison à partir des k parents de la liste X, qui calcule la moyenne des coordonnées pour chaque élément des vecteurs latents, avec une probabilité p pour chaque élément
        Sinon, on choisit un élément parmi les k vecteurs
    Args :
        X (np.array): Liste de vecteurs de l'espace latent, de longueur k
        p (float) : probabilité de modifier l'élément à chaque tour de boucle (p entre 0 et 1, si p vaut 1, new_vector est la moyenne de X1 et X2, si p vaut 0, les éléments de new_vector sont extraits soit de X1, soit de X2)
    Return :
        new_vector(np.array) : np.array formant un nouveau vecteur de l'espace latent, recombinaison des k parents'''
    k = len(X)
    new_vector = []
    n = len(X[0])
    for i in range(n):
        if np.random.rand() < p :
            x = 0
            for j in range(k):
                x += X[j][i]
            new_vector.append(x/k)
        else :
            rand = np.random.rand()
            proba = 1/k
            for j in range(k):
                if rand <= proba :
                    new_vector.append(X[j][i])
                    break
                proba += 1/k
    return np.array(new_vector)



def one_iteration_mutation(List_parents, type_evol, n, step_size) :
    '''This function calculates the next generation of children from a list of parents. The children are generated only by mutation here, according to the type of evolution str_images_chosen
    Args :
        List_parents (np.array/list) : list of parents that we want to generate children the next generation
        type_evol (str) : type of evolution, '+' if we want the parent to be part of the next generation, ',' otherwise
        n (int) : number of children we want to be generated from each parents
        step_size (float) : parameter of mutation
    Return :
        List (np.array) of latent vectors that are the children of the parents in List_parents, length should be n*len(List_parents) if type_evol == ',', and (n+1)*len(List_parents) otherwise'''
    L_children = []
    for parent in List_parents :
        if type_evol == '+':
            Parent = generation_copies(parent, n)
            Children = mutation(Parent, step_size, 1)
            Children = np.vstack([Children, parent])
            for element in Children :
                L_children.append(element)
        else :
            Parent = generation_copies(parent, n)
            Children = mutation(Parent, step_size, 1)
            for element in Children :
                L_children.append(element)
    return np.array(L_children)


def adjust_children(List_parents, prior):
    '''This function creates a list of 10 children from List_parents, thanks to the function mutation et recombination, depending on len(List_parents)
        It also adds to the children generated from the parents random images generated by the autoencoder, so that the user can always change the direction of search of the algorithm
    Args:
        List_parents (np.array) : List of the images selected by the user from which we want to generate new images
        prior : previously constructed for the neural network, necessary to generate new images from the autoencoder
    Return:
        L (np.array) :
        '''
    n = len(List_parents)
    if n == 1:
        L = one_iteration_mutation(List_parents, ',', 4, 1.5)
        L = np.vstack([L, one_iteration_mutation(List_parents, '+', 3, 1)])
        #print(L)
        new_images = generate_latent_vectors(prior, 2)
        L = np.vstack([L, new_images])
        #print(L)
        #random.shuffle(L)
        return L
    elif n == 2 :
        L = one_iteration_mutation(List_parents, ',', 3, 1.5)
        new_images = generate_latent_vectors(prior, 2)
        L = np.vstack([L, new_images])
        recombination1 = recombination_k_parents(List_parents, 1)
        recombination2 = recombination_k_parents(List_parents, 0)
        L = np.vstack([L, recombination1])
        L = np.vstack([L, recombination2])
        return L
    elif n == 3 :
        L = one_iteration_mutation(List_parents, ',', 2, 1.5)
        new_images = generate_latent_vectors(prior, 2)
        L = np.vstack([L, new_images])
        recombination1 = recombination_k_parents(List_parents, 1)
        recombination2 = recombination_k_parents(List_parents, 0)
        L = np.vstack([L, recombination1])
        L = np.vstack([L, recombination2])
        return L
    elif n == 4 :
        L = one_iteration_mutation(List_parents, ',', 1, 1.5)
        new_images = generate_latent_vectors(prior, 2)
        L = np.vstack([L, new_images])
        recombination1 = recombination_k_parents(List_parents, 1)
        recombination2 = recombination_k_parents(List_parents, 0)
        recombination3 = recombination_k_parents([List_parents[0], List_parents[1]], 0)
        recombination4 = recombination_k_parents([List_parents[2], List_parents[3]], 1)
        L = np.vstack([L, recombination1])
        L = np.vstack([L, recombination2])
        L = np.vstack([L, recombination3])
        L = np.vstack([L, recombination4])
        return L




## Main program
if __name__ == "__main__":
    X = np.array([[1.0,2.0,3.0],[2.0,5.0,9.0]])
    #print(X)
    #Y = mutation(X,1, 1)
    #print(Y)
    X1 = np.array([1.0,2.0,3.0])
    X2 = np.array([5.0,6.0,7.0])
    X3 = recombination_2parents_moy(X1, X2, 1)
    #print(X3)
    X4 = recombination_k_parents(X, 0)
    #print(X4)
    Y = one_iteration_mutation([X1], '+', 3, 1)
    print(Y)


#Faire des units tests
#Faire la fonction selection_function
#faire la documentation
#mettre une probabilité de mutations ?
#Penser à rajouter des images aléatoires et à modifier le step-size
#Utiliser les attributs ?
