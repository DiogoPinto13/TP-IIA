#Diogo Pinto 2020133653
#Francisco Almeida 2020138795
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:38:01 2022

@author: ratking
"""
from collections import defaultdict
from http.client import OK
from itertools import count
from secrets import randbits
from tabnanny import check
import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import io
import threading
import multiprocessing 
import os
import time
from numpy import random
import numpy as np

# VARIAVEIS GLOBAIS
G = nx.Graph()
graph = defaultdict(list)
solucoes = []
Pros = []
# FUNCOES
# LEITURA DO FICHEIRO
def read_file(filename, n_iter, popsize, r_cross, r_mut):
    print("Reading from file " + filename)
    n_vert = 0
    with open(filename, "r+") as file:
        for line in file:
            if line.startswith("e"):
                x = line.split(" ")
                graph[int(x[1])].append(int(x[2]))
                graph[int(x[2])].append((int(x[1])))
            elif line.startswith("p"):
                x = line.split(" ")
                n_vert = int(x[2])
                
    matrix = convert_matriz_adj(graph)
    #solucao, carndinalidade= genetic_algo_hibrido1(n_vert, n_iter, popsize, r_cross, r_mut,matrix)

    #solucao, carndinalidade= genetic_algo(n_vert, n_iter, popsize, r_cross, r_mut,matrix)

    solucao = hibrido2(n_vert,n_iter,popsize,r_cross,r_mut,matrix)
    
    #solucao = trepa_colinas(solucoes,n_iter,graph,n_vert)


    print("A cardinalidade" + str(filename) + " e: ")
    #print(carndinalidade)

def generate_sol2(graph,bounds):
    num_guardados= []
    contador = 1
    while(contador<=bounds[1]*2): 
        flag = 0
        key = rd.randint(bounds[0],bounds[1])
        if key not in num_guardados:
            for m in graph[key]:
                if m in num_guardados:
                    flag = 1
                    contador-=1
                    break
            if not flag: 
                num_guardados.append(key)
        contador += 1
    return num_guardados

def generate_sol1(graph,bounds):
    num_guardados= []
    for j in range(bounds[0],bounds[1]*20): 
        flag = 0
        key = rd.randint(bounds[0],bounds[1])
        if key not in num_guardados:
            for m in graph[key]:
                if m in num_guardados:
                    flag = 1
                    break
            if not flag: 
                num_guardados.append(key)
    return num_guardados

#algoritmo genetico    
def convert_matriz_adj(graph):
    
    matrix = [[0 for j in range(len(graph))]
                 for i in range(len(graph))]

    for i in range(1,len(graph)+1):
        for j in graph[i]:
            matrix[i-1][j-1] = 1

    
    return matrix
    

def gera_pop(popsize,num_gen,n_verts):
    individual = []
    temp = []
    i = 0
    while(popsize>0):
        for j in range(0,num_gen):
            temp.append(rd.randint(1,n_verts))
        if check_sol(temp):
            individual.append(temp[:])
            #print(individual)
            popsize -= 1
        temp.clear()
        i+=1
    #list2 = [x for x in individual if x != []]
    return individual


def torneio(popsize,pop):
    parents = []
    for i in range(0,popsize):
        x1 = pop[rd.randint(0,popsize-1)]
        x2 = pop[rd.randint(0,popsize-1)]
        while(x2 == x1):
            x2 = pop[rd.randint(0,popsize-1)]
        if len(x1) > len(x2):
            parents.append(x1)
        else:
            parents.append(x2)
    return parents

def gera_pop_matrix(popsize,n_verts,matrix):
    pop = [rd.randint(0, 2, n_verts).tolist() for _ in range(popsize)]
        #pop[i][rd.randint(0,n_verts-1)] = rd.randint(0,1)
        #pop.append(matrix[(rd.randint(0,n_verts-1))])
    return pop



def trepa_colinas(solucoes,n_iter,graph,nvertices):
    for k in range(1,n_iter):
        
        #print("Iteracao num :" + str(k))
        num_guardados= generate_sol1(graph,[1,nvertices])
        custo = len(num_guardados)
        #print("CUSTO " + str(custo))
        if(custo >= len(solucoes)):
            solucoes.clear()
            for num in num_guardados:
                solucoes.append(num)
        num_guardados.clear()
    return solucoes

def convert_numeros(sol_a_converter,tam_matrix):
    convert = np.zeros(tam_matrix,dtype=int).tolist()
    #print(sol_a_converter)
    for i in sol_a_converter:
        convert[i-1] = 1
    return convert
    
def check_sol(solucao):
    for key in solucao:
        if solucao.count(key) > 1:
            return False
        for adj in graph[key]:
            if adj in solucao:
                return False
    return True

def check_sol_matriz(sol,matrix):
    contador = 0
    #print("SOL LEN : "+str(len(sol)) + "SOL_SETLEN: " +str(sol.count(0)))
    #print(sol)
    #if not any(sol):     
        #return False
    for i in range(0,len(sol)):
        if(sol[i] == 0):
            continue
        for j in matrix[i]:
            #print(contador)
            if j == 1 and sol[contador] == 1:
                return False
            else:
                contador += 1
                continue
        contador = 0
    return True
                
def mutation(pop,prob_mutacao):
    for i in range(0,len(pop)):
        for j in pop[i]:
            if rd.random() < prob_mutacao:
                j = 1 - j
    return pop

def printMatrix(adj):
     
    for i in range(len(adj)):
        for j in range(len(adj)):
            print(adj[i][j], end = ' ')

        print()

    print()

def comeca(n_iter, popsize, r_cross, r_mut):
    #read_file("file1.txt", n_iter, popsize, r_cross, r_mut)
    for i in range(1,7):
        nomeFich = "file" + str(i) + ".txt"
        p1 = multiprocessing.Process(target=read_file, args=(nomeFich, n_iter, popsize, r_cross, r_mut))
        Pros.append(p1)
        p1.start()
        #p1.join()
        
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = random.randint(len(pop))
    for ix in random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def objective(c,matrix):
    if check_sol_matriz(c,matrix):
        return c.count(1)
    else:
        return 0
def mutation3(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

def mutationKing(solucao, n_mutacao):
    if random.rand() < n_mutacao:
        for i in range(0,int(len(solucao)/2)):
            solucao[i] = 1
        for j in range(int(len(solucao)/2),len(solucao)):
            solucao[j] = 0
        rd.shuffle(solucao)

def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = random.randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]

    return [c1, c2]

def crossover2(p1, p2, r_cross):

    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt1 = random.randint(1, len(p1)-2)
        pt2 = random.randint(pt1,len(p1)-2)
        
        c1 = p1[:pt1] + p2[pt1:pt2] + p1[pt2:]
        c2 = p2[:pt1] + p1[pt1:pt2] + p2[pt2:]

    return [c1, c2]
def cria_individ(popsize,n_verts,matrix):
    #print(n_verts)
    individ = []
    individ = np.random.randint(0,1,n_verts).tolist()
    while(check_sol_matriz(individ,matrix) == False):
        individ = np.random.randint(0,1,n_verts).tolist()
    #print("Criou individo com cardinalidade: " + str(individ.count(1)))
    return individ

def genetic_algo(n_verts, n_iter, popsize, r_cross, r_mut,matrix):
    pop = [cria_individ(popsize,n_verts,matrix) for _ in range(popsize)]
    best, best_eval = 0, 0
    for gen in range(n_iter):
        scores = [objective(c,matrix) for c in pop]
        for i in range(popsize):
            if scores[i] > best_eval:                                                
                best, best_eval = pop[i], scores[i]
                #print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        selected = [selection(pop, scores) for _ in range(popsize)]
        children=list()
        for i in range(0, popsize, 2):
            
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover2(p1, p2, r_cross):
                # mutation
                #mutationKing(c,r_mut)
                mutation3(c, r_mut)
                # store for next generation
                children.append(c)
        pop = children
    return [best, best_eval]


def cria_individ_hibrido(popsize,n_verts):
    individ = []
    individ = convert_numeros(generate_sol2(graph,[0,n_verts]),n_verts)
    #print("Criou individo com cardinalidade: " + str(individ.count(1)))
    return individ

def genetic_algo_hibrido1(n_verts, n_iter, popsize, r_cross, r_mut,matrix):
    #pop = [cria_individ(popsize,n_verts,matrix) for _ in range(popsize)]
    pop = [cria_individ_hibrido(popsize,n_verts) for _ in range(popsize)]
    best, best_eval = 0, 0
    for gen in range(n_iter):
        scores = [objective(c,matrix) for c in pop]
        for i in range(popsize):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
        selected = [selection(pop, scores) for _ in range(popsize)]
        children=list()
        for i in range(0, popsize, 2):
            
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation3(c, r_mut)
                # store for next generation
                children.append(c)
        pop = children
    return [best, best_eval]
def convert_bin(convert):
    res = []
    contador=1
    for i in convert:
        if i == 1:
            res.append(contador)
        contador += 1
    #print(res)
    return res

def hibrido2(n_verts,n_iter,popsize,r_cross,r_mut,matrix):
    result, cardinalidade = genetic_algo(n_verts,n_iter,popsize,r_cross,r_mut,matrix)
    #print(result)
    if result != 0:
        result = convert_bin(result)
    else:
        result = []
    #print(cardinalidade)
    t = trepa_colinas(result,n_iter,graph,n_verts)
    #print(t)
    print("Cardinalidade: " + str(len(t)))

#------------MAIN--------------          
if __name__ == "__main__":   

    starttime = time.time()
    n_iter = 2500
    popsize = 100
    r_cross = 0.7
    r_mut = 0.05
    comeca(n_iter, popsize, r_cross, r_mut)
    for t in Pros:
        t.join()
    print('O programa demorou {} segundos'.format(time.time() - starttime))