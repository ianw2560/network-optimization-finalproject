import json
import numpy as np
import sys
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
from scipy.stats import chi2_contingency

def create_graph(in_list, in_weight, out_list, out_weight, username_list, party_list):

	G = nx.DiGraph()

	for i in range(len(username_list)):
		G.add_node(i, username=username_list[i], party=party_list[i])

	for i in range(len(in_list)):
		for j in range(len(in_list[i])):
			G.add_edge(in_list[i][j], i, weight=in_weight[i][j])
			
			
	for i in range(len(out_list)):
		for j in range(len(out_list[i])):
			G.add_edge(i, out_list[i][j], weight=out_weight[i][j])

	return G

def pagerank(graph, username_list, df=0.85, eps=1e-6):

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

	scores = []

	for i in range(len(ranking)):

		# Round the scores to 6 decimal places
		scores.append( np.round(R[ranking[i]], 6) )

	scores = np.array(scores)

	return ranking, scores

def leader_comparison(subgroup, G, username_list):

	# Set isLeader to 0
	for i in range(len(G.nodes)):
		isleader = {i: {"isLeader": 0}}
		nx.set_node_attributes(G, isleader)

	# Set the nodes in leadersAll to 1
	for i in range(len(subgroup)):
		isLeader = { subgroup[i] : {"isLeader": 1} }
		nx.set_node_attributes(G, isLeader)

	graph = nx.to_numpy_array(G)

	#=============================================
	# Calculate PageRank rankings
	#=============================================

	# Calculate the PageRank scores
	rankings, scores = pagerank(graph, username_list)

	# Print the results
	print("PageRank Rankings")
	print("=================")
	for i in range(len(rankings)):
		# Print the rankings in the requested format
		print("Rank " + str(i) + ":", username_list[rankings[i]], str(rankings[i]) +" ("+ str(scores[i]) +")")
	print()

	print("======================================")
	print("              ANALYSIS")
	print("======================================")
	print()

	#=============================================
	# Create Pandas dataframe with leader category
	#=============================================

	isLeader = []
	for i in range(len(G.nodes)):
		l = nx.get_node_attributes(G, "isLeader")
		isLeader.append(l[rankings[i]])

	# Create a Pandas DataFrame
	df = pd.DataFrame({'PageRank_Rank': np.arange(1, len(rankings) + 1), 'Node': rankings, 'isLeader': isLeader, "Score": scores})
	df.set_index('Node', inplace=True)

	#==================================
	# Perform chi-square test
	#==================================

	# Discretize the ranks into four quartiles
	rank_percentiles = pd.qcut(df['PageRank_Rank'], q=10)
	contingency_table = pd.crosstab(df['isLeader'], rank_percentiles)

	# Perform the Chi-Square test
	res = chi2_contingency(contingency_table)

	print("Perform chi-square test")
	print("=========================")

	# Print p-value
	print("P-value:", res.pvalue)

	# Interpret the p-value
	alpha = 0.05
	if res.pvalue < alpha:
		print("There is a significant association between being a leader and having a higher PageRank rank.")
	else:
		print("There is no significant association between being a leader and having a higher PageRank rank.")
	print()

	#===================================
	# Print the PageRanks of the leaders
	#===================================

	# Get a DataFrame with only the leaders
	leader_nodes_df = df[df['isLeader'] == 1]

	print("PageRanks of Leaders")
	print("=========================")

	print(leader_nodes_df)
	print()

	#===================================
	# Print descriptive statistics
	#===================================

	print("Descriptive Statistics for All Members (PageRank_Rank):")
	print("=======================================================")
	print(df['PageRank_Rank'].describe())
	print()

	# Display the descriptive statistics
	print("Descriptive Statistics for Leaders (PageRank_Rank):")
	print("===================================================")
	print(leader_nodes_df['PageRank_Rank'].describe())
	print()

	print("Descriptive Statistics for All Members (Scores):")
	print("================================================")
	print(df['Score'].describe())
	print()

	cols = list(df.columns)
	cols.remove('PageRank_Rank')
	cols.remove('isLeader')
	df[cols]

	for col in cols:
		col_zscore = col + '_zscore'
		df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)

	print("Z-Scores for PageRank Scores for Leaders")
	print("========================================")
	print("Node \tZ-Score")
	for leader in subgroup:
		zscore = df.loc[leader]['Score_zscore']
		print(leader, "\t{:.5f}".format(zscore))

	# Plot the scores of all the nodes
	plot_scores(rankings, scores, subgroup)

# Create a graph of members vs scores
def plot_scores(rankings, scores, leaders):

	leader_indexes = []
	leader_scores = []

	for leader in leaders:
		leader_indexes.append(np.where(rankings==leader)[0][0])
	
	for index in leader_indexes:
		leader_scores.append(scores[index])

	plt.figure(figsize=(8, 6))

	plt.scatter(rankings, scores)
	plt.scatter(leaders, leader_scores, c='red', label='Leaders')

	plt.title('PageRank Scores of Congressional Leaders')
	plt.ylabel('PageRank Scores')
	plt.xlabel('Member ID')

	plt.grid(True)
	plt.legend()

	plt.savefig('results/pagerank_scores.png')

# Read in network data
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

# Create a graph from the edgelist file
G = nx.read_weighted_edgelist('congress.edgelist', nodetype=float, create_using=nx.DiGraph)
G = create_graph(in_list, in_weight, out_list, out_weight, username_list, party_list)

# Add username attributes to all nodes and set isLeader to 0
for i in range(len(G.nodes)):
	# print(i, username_list[i])
	usernames = {i: {"username": username_list[i]}}
	nx.set_node_attributes(G, usernames)

# The node IDs of top ten House/Senate leadership positions
leaders = [367, 71, 254, 322, 48, 25, 160, 80, 399]

leader_comparison(leaders, G, username_list)
