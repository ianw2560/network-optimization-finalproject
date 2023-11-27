import json
import numpy as np
import sys
from matplotlib import pyplot as plt
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import SpectralClustering

def create_undirected_graph(in_list, out_list, username_list, party_list):

	G = nx.Graph()

	for i in range(len(username_list)):
		G.add_node(i, username=username_list[i], party=party_list[i])

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

	for i in range(iterations):

		# Expansion Step
		expanded_matrix = np.linalg.matrix_power(transition_matrix, 2)
		
		# Inflation Step
		inflated_matrix = np.power(expanded_matrix, inflation)
		
		# Normalization Step
		sum_columns = np.sum(inflated_matrix, axis=0, keepdims=True)
		transition_matrix = inflated_matrix / sum_columns

	# Identify clusters based on rows or columns of the final matrix
	clusters = np.argmax(transition_matrix, axis=0)

	# Change the unqiue values of the final transition matrix to integers starting from 0
	cluster_labels = np.unique(clusters)

	for i in range(len(cluster_labels)):
		for j in range(len(clusters)):
			if clusters[j] == cluster_labels[i]:
				clusters[j] = i
	
	return clusters

def get_mcl_scores(labels_true):

	inflation_value = 1.6
	increment = 0.025
	mcl_scores = []

	num_inflation_values = int((2 - inflation_value) /increment)

	for i in range(num_inflation_values):

		inflation_value = np.round(inflation_value + increment, 3)
		print("Current Inflation Value:", inflation_value)

		# Run MCL on the network with the current inflation value
		mcl_labels_pred = markov_clustering(G, inflation=inflation_value)

		# Get the number of clusters that MCL created
		num_clusters = len(np.unique(mcl_labels_pred))

		if num_clusters == 2:

			# Run NMI and AMI on the labels predicted by MCL
			nmi_score = normalized_mutual_info_score(mcl_labels_pred, labels_true)
			ami_score = adjusted_mutual_info_score(mcl_labels_pred, labels_true)
			ari_score = adjusted_rand_score(mcl_labels_pred, labels_true)

			mcl_scores.append( (inflation_value, np.round(nmi_score, 5), np.round(ami_score, 5), np.round(ari_score, 5)) )

	return mcl_scores

f = open('congress_network_data.json')
data = json.load(f)

in_list = data[0]['inList']
in_weight = data[0]['inWeight']
out_list = data[0]['outList']
out_weight = data[0]['outWeight']
username_list = data[0]['usernameList']
party_list = data[0]['partyList']


# Get true labels for political party affiliations
labels_true = np.array(party_list)
for i in range(len(labels_true)):
	if labels_true[i] == 'D':
		labels_true[i] = 1
	elif labels_true[i] == 'R':
		labels_true[i] = 0

labels_true = np.array(labels_true, dtype=int)

# Create graph for MCL and SC
G = create_undirected_graph(in_list, out_list, username_list, party_list)
G = nx.to_numpy_array(G, dtype=int)

sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100, assign_labels='discretize')
sc_labels_pred = sc.fit_predict(G)  

sc_nmi = np.round( normalized_mutual_info_score(sc_labels_pred, labels_true) , 5)
sc_ami = np.round( adjusted_mutual_info_score(sc_labels_pred, labels_true) , 5)
sc_ari = np.round( adjusted_rand_score(sc_labels_pred, labels_true) , 5)

mcl_scores = get_mcl_scores(labels_true)
print(mcl_scores)

print("MCL Scores")
print("==========")

for score in mcl_scores:
	print("[Inflation =", str(score[0]), "]\tNMI:", str(score[1]), "AMI:", str(score[2]), "ARI:", str(score[3]))

print("Spectral Clustering Scores")
print("==========================")
print("NMI:", sc_nmi)
print("AMI:", sc_ami)
print("ARI:", sc_ari)


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
