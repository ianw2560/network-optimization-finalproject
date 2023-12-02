import json
import numpy as np
import sys
from matplotlib import pyplot as plt
import networkx as nx

def create_graph(in_list, in_weight, out_list, out_weight, username_list, party_list):

	G = nx.DiGraph()

	for i in range(len(username_list)):
		G.add_node(i, username=username_list[i], party=party_list[i])

	for i in range(len(in_list)):
		for j in range(len(in_list[i])):
			G.add_edge(i, in_list[i][j])
			
			
	for i in range(len(out_list)):
		for j in range(len(out_list[i])):
			G.add_edge(i, out_list[i][j])


	for i in range(len(in_weight)):
		for j in range(len(in_weight[i])):
			nx.set_edge_attributes(G, {(i, j) : {"weight": in_weight[i][j] } })

	for i in range(len(out_weight)):
		for j in range(len(out_weight[i])):
			nx.set_edge_attributes(G, {(i, j) : {"weight": out_weight[i][j] } })

	return G

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

# Create a graph from the JSON data
G = nx.read_weighted_edgelist('congress.edgelist', nodetype=float, create_using=nx.DiGraph)

# G = create_graph(in_list, in_weight, out_list, out_weight, username_list, party_list)

# Add username attributes to all nodes and set isLeader to 0
for i in range(len(G.nodes)):
	usernames = {i: {"username": username_list[i]}}
	isleader = {i: {"isLeader": 0}}
	nx.set_node_attributes(G, usernames)
	nx.set_node_attributes(G, isleader)


# The node IDs of top ten House/Senate leadership positions
leadersAll = [367, 71, 254, 322, 48, 25, 160, 80, 399]

# The node IDs of top House/Senate leadership positions
leadersTop = [367, 71, 254, 322]


# Set the nodes in leadersAll to 1
for i in range(len(leadersAll)):
	isLeader = { leadersAll[i] : {"isLeader": 1} }
	nx.set_node_attributes(G, isLeader)

# print(G.get_edge_data(0, 4)['weight'])

graph = nx.to_numpy_array(G)

print(graph.shape)
print(graph)
# print(graph[0:4][0])
# print(graph[0][3])
print(graph[0][76])
print(graph[4][0])
# print(graph[334][473])

rankings = pagerank(graph)



# print(G.nodes[0]["isLeader"])
# print(G.nodes[1]["isLeader"])

# print(G.nodes[0]["username"])

# for node in G.nodes:
# 	print(str(node) + ": " + G.nodes[node]["username"])
# print(G)

# for i in range(len(G.nodes)):
# 	print(G.nodes[i]["username"])
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
