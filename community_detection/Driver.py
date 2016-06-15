import plotly.tools as tls
tls.set_credentials_file(username='darkaeon10', api_key='qmdqrxf4mc')

from igraph import *
import plotly.plotly as py
from plotly.graph_objs import *

from tweets import TweepyHelper
from tweets import TweetUtils

# g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)])
# g.vs["name"] = ["Alice", "Bob", "Claire", "Dennis", "Esther", "Frank", "George"]
# g.vs["age"] = [25, 31, 18, 47, 22, 23, 50]
# g.vs["gender"] = ["f", "m", "f", "m", "f", "m", "m"]
# g.es["is_formal"] = [False, False, True, True, True, False, True, False, False]
# plot(g)

# print(TweepyHelper.api.followers_ids(TweepyHelper.api.me()))

G = TweetUtils.TweetUtils().construct_follow_graph(None, TweepyHelper.api.me().id, 50, False)

labels = list(G.vs['name'])
N = len(labels)
E= [e.tuple for e in G.es]

layt = G.layout('graphopt')
community = G.community_multilevel().membership
communities = len(set(community))

# dendrogram = g.community_edge_betweenness()
# clusters = dendrogram.as_clustering()
# summary(g)
# plot(g)
# plot(clusters)

color_list = ['#6959CD', '#DD5E34', '#69CD45', '#000005']

Xn = [layt[k][0] for k in range(N)]
Yn = [layt[k][1] for k in range(N)]
Xe = []
Ye = []
for e in E:
    Xe += [layt[e[0]][0], layt[e[1]][0], None]
    Ye += [layt[e[0]][1], layt[e[1]][1], None]

trace1 = Scatter(x=Xe,
               y=Ye,
               mode='lines',
               line=Line(color='rgb(210,210,210)', width=1),
               hoverinfo='none'
               )

node_x = [[] for i in range(communities)]
node_y = [[] for i in range(communities)]
labelz = [[] for i in range(communities)]

for j in range(len(community)):
	index = community[j]
	node_x[index].append(layt[j][0])
	node_y[index].append(layt[j][1])
	labelz[index].append(labels[j])

plot_data = [trace1]
for i in range(communities):
	trace = Scatter(x=node_x[i],
					y=node_y[i],
					mode='markers',
					name='ntw',
					marker=Marker(symbol='dot',
								  size=5,
								  color=color_list[i],
								  line=Line(color='rgb(50,50,50)', width=0.5)
								  ),
					text=labelz[i],
					hoverinfo='text'
					)

	plot_data.append(trace)

axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

width = 800
height = 800

layout = Layout(title= "GitHub Network",
    font=Font(size=12),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=XAxis(axis),
    yaxis=YAxis(axis),
    margin=Margin(
        l=100,
        r=40,
        b=85,
        t=100,
    ),
    hovermode='closest',
    annotations=Annotations([
           Annotation(
           showarrow=False,
            text='This igraph.Graph has the graphopt layout',
            xref='paper',
            yref='paper',
            x=0,
            y=-0.1,
            xanchor='left',
            yanchor='bottom',
            font=Font(
            size=14
            )
            )
        ]),
    )

data = Data(plot_data)
fig = Figure(data=data, layout=layout)
py.plot(fig, filename='github-network-community-igraph')


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
