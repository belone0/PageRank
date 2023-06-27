import numpy as np

def pagerank(relations, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    #fator 0.85, limite 100 iteracoes

    #determina o numero de empresas com base no tamnhano do array de relacoes e cria matriz
    num_companies = len(relations)
    adjacency_matrix = np.zeros((num_companies, num_companies))

    #preenchendo matriz estocastica com as relacoes
    for i in range(num_companies):
        total_relations = len(relations[i])
        if total_relations == 0:
            adjacency_matrix[:, i] = 1 / num_companies
        else:
            for j in range(num_companies):
                if i != j:
                    adjacency_matrix[j, i] = relations[i].count(j) / total_relations

    # normalizando a soma das colunas para 1
    column_sums = adjacency_matrix.sum(axis=0)
    column_sums[column_sums == 0] = 1  #evita divisao por 0
    adjacency_matrix /= column_sums[np.newaxis, :]

    #criacao da matriz de probabilidade
    teleportation = np.full((num_companies, num_companies), 1 / num_companies)

    
    #fazendo conta G = a*A + (1-a)*P
    #A = matriz est, P = matriz porcentagens
    #a = fator de multiplicacao    
    matrix = damping_factor * adjacency_matrix + (1 - damping_factor) * teleportation

    #inicialização dos ranks
    ranks = np.ones(num_companies) / num_companies

    #matriz multiplicada diversas vezes até atingit equilibrio (mudança menor que a tolerancia)
    for _ in range(max_iterations):
        new_ranks = np.dot(matrix, ranks)
        if np.linalg.norm(new_ranks - ranks) < tolerance:
            break
        ranks = new_ranks

    #retorna a matriz adjacente e os ranks finais
    return adjacency_matrix, ranks

# prompts
num_companies = int(input("Enter the number of companies: "))
relations = []
for i in range(num_companies):
    num_relations = int(input(f"Enter the number of relations for Company {i}: "))
    company_relations = []
    for _ in range(num_relations):
        relation = int(input(f"Enter the companie that Company {i} relates to: "))
        company_relations.append(relation)
    relations.append(company_relations)

#chamada da funcao
adjacency_matrix, ranks = pagerank(relations)

#show
print("Adjacency Matrix:")
print(adjacency_matrix)
print()

#scores
print("Company Scores:")
ranked_companies = np.argsort(ranks)[::-1]
for rank, company in enumerate(ranked_companies, start=1):
    print(f"Rank {rank}: Company {company} (PageRank: {ranks[company]})")
