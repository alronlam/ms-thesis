from igraph import *

g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]
plot(g)


# import igraph
# import csv

# g = igraph.Graph.Read_Ncol('network.txt')

# g = igraph.Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
# g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
# g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
# g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
# g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]
#
# igraph.plot(g)
#
# dendrogram = g.community_edge_betweenness()
# clusters = dendrogram.as_clustering()
#
# membership = clusters.membership
#
# igraph.plot(clusters)


# with open('output.csv', 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file, delimiter =' ')
#
#     for name, membership in zip(g.vs["name"], membership):
#         writer.writerow([name, membership])
