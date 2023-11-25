import json
import numpy as np
import sys
from matplotlib import pyplot as plt
import networkx as nx

# f = open('congress_network_data.json')
# data = json.load(f)

# inList = data[0]['inList']
# inWeight = data[0]['inWeight']
# outList = data[0]['outList']
# outWeight = data[0]['outWeight']
# usernameList = data[0]['usernameList']


def parseGraph(filename):

	"""
	Returns a numpy array of the graph
	"""
	
	file = open(filename)

	graph = []
	for line in file:

		# Remove the first character, whitespace, 
		# and split numbers into an array
		line = line.split('\t')
		line = line[1].split(',')
		
		# Convert string array into ints
		line = [int(x) for x in line]

		graph.append(np.array(line))

	return np.array(graph)

def pagerank(graph, df=0.85, eps=1e-5):

	# Specify the number of iterations
	num_iterations = 100

	# Let N be the number of nodes in the graph
	N = graph.shape[0]

	# Initialize the vector R to 1/N
	R = np.ones(N) / N

	# Normalize the S matrix that will be multiplied by R
	S = graph / graph.sum(axis=1, keepdims=True)
	S = S.T

	# Loop for num_iterations
	for i in range(num_iterations):

		# Calcualte the next R vector
		R_next = (1 - df) / N + df * np.matmul(S, R)

		# Check for covergence by taking the sums of the magnitude
        # of R_next - R and seeing if it is less than epsilon.
        # If it is less than epsilon, we terminate the loop early
		diff = R_next - R

		# Take the sum of the magnitude of the difference
		convergence = np.sum(np.sqrt(diff.dot(diff)))

		# Check if this value is less than epsilon
		if convergence < eps:
			break

		# Set R to R_next
		R = R_next

	# Sort nodes by PageRank score (highest to lowest)
	ranking = np.argsort(R)[::-1]

	# Print the results
	print("ranking:", end=" ")
	for i in range(len(ranking)):

		# Round the scores to 6 decimal places
		score = np.round(R[ranking[i]], 6)

		# Print the rankings in the requested format
		if i < len(ranking) - 1:
			print(str(ranking[i]) +" ("+ str(score) +"), ", end="")
		else:
			print(str(ranking[i]) +" ("+ str(score) +")")


def create_undirected_graph(in_list, out_list):

	G = nx.Graph()

	for i in range(len(in_list)):
		for j in range(len(in_list[i])):
			#print("adding edge", str(in_list[i][j]), str(i))
			G.add_edge(i, in_list[i][j])
			

	for i in range(len(out_list)):
		for j in range(len(out_list[i])):
			#print("adding edge", str(i), str(in_list[i][j]))
			G.add_edge(i, out_list[i][j])

	return G


def markov_clustering(adjacency_matrix, inflation=2, iterations=100):
    # Normalize the adjacency matrix to create a transition matrix
    transition_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0, keepdims=True)
    
    for _ in range(iterations):

        # Expansion Step
        expanded_matrix = np.linalg.matrix_power(transition_matrix, 2)
        
        # Inflation Step
        inflated_matrix = np.power(expanded_matrix, inflation)
        
        # Normalization Step
        sum_columns = np.sum(inflated_matrix, axis=0, keepdims=True)
        transition_matrix = inflated_matrix / sum_columns
        
    # Identify clusters based on rows or columns of the final matrix
    clusters = np.argmax(transition_matrix, axis=0)

	cluster_list = []

	for
    
    return clusters


f = open('congress_network_data.json')
data = json.load(f)

in_list = data[0]['inList']
in_weight = data[0]['inWeight']
out_list = data[0]['outList']
out_weight = data[0]['outWeight']
username_list = data[0]['usernameList']


G = create_undirected_graph(in_list, out_list)
G = nx.to_numpy_array(G)

clusters = markov_clustering(G)

print(clusters)


# print(type(inList))

# print(len(inList))
# print(len(outList))


# Create a graph from the JSON data
#G = nx.read_weighted_edgelist('congress.edgelist', nodetype=int, create_using=nx.DiGraph)

##################
# Draw graph
# pos = nx.spring_layout(G)  # You can use different layout algorithms
# fig, ax = plt.subplots(figsize=(20, 20))
# nx.draw(G, pos, with_labels=False, font_size=10, font_color='black', node_color='skyblue', edge_color='black')

# # Draw the node labels
# # node_labels = {node: str(node) for node in G.nodes()}
# # nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=, font_color='black', font_weight='bold', ax=ax)

# plt.savefig("graph_image.png", dpi=500)

##################

# Get the weight of a specific edge
source_node = 0
target_node = 86

print(username_list[87])

# Check if the edge exists in the graph
# if G.has_edge(source_node, target_node):
#     edge_data = G.get_edge_data(source_node, target_node)
#     #weight = edge_data['weight']
#     print(f"The weight of the edge ({source_node}, {target_node}) is: {weight}")
# else:
#     print(f"The edge ({source_node}, {target_node}) does not exist in the graph.")

# print(G.get_edge_data(4, 0))

# print(G.get_edge_data(4, 0)['weight'])

# create_undirected_graph(in_list, in_weight, out_list, out_weight)

# for n in inList:
# 	print(n)

