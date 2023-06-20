import numpy as np

def pagerank(votes, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_guys = len(votes)
    adjacency_matrix = np.zeros((num_guys, num_guys))

    for i in range(num_guys):
        total_votes = len(votes[i])
        if total_votes == 0:
            adjacency_matrix[:, i] = 1 / num_guys
        else:
            for j in range(num_guys):
                if i != j:
                    adjacency_matrix[j, i] = votes[j].count(i) / total_votes

    # Normalize the columns to sum to 1
    column_sums = adjacency_matrix.sum(axis=0)
    adjacency_matrix /= column_sums[np.newaxis, :]

    teleportation = np.full((num_guys, num_guys), 1 / num_guys)
    matrix = damping_factor * adjacency_matrix + (1 - damping_factor) * teleportation
    ranks = np.ones(num_guys) / num_guys

    for _ in range(max_iterations):
        new_ranks = np.dot(matrix, ranks)
        if np.linalg.norm(new_ranks - ranks) < tolerance:
            break
        ranks = new_ranks

    return adjacency_matrix, ranks

# Prompt for number of guys and their votes
num_guys = int(input("Enter the number of guys: "))
votes = []
for i in range(num_guys):
    num_votes = int(input(f"Enter the number of votes for Guy {i}: "))
    guy_votes = []
    for _ in range(num_votes):
        vote = int(input(f"Enter the guy voted for by Guy {i}: "))
        guy_votes.append(vote)
    votes.append(guy_votes)

# Calculate PageRank
adjacency_matrix, ranks = pagerank(votes)

# Print the adjacency matrix
print("Adjacency Matrix:")
print(adjacency_matrix)
print()

# Print the PageRank scores
print("PageRank Scores:")
for guy, rank in enumerate(ranks):
    print(f"Guy {guy}: {rank}")