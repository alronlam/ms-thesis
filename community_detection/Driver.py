import plotly.tools as tls
tls.set_credentials_file(username='darkaeon10', api_key='qmdqrxf4mc')

from igraph import *
import plotly.plotly as py
from plotly.graph_objs import *
from datetime import datetime

from tweets import TweepyHelper
from tweets import TweetUtils

from database import DBManager

# DBManager.delete_followers_ids(461053984)
# DBManager.delete_following_ids(461053984)
# print(DBManager.get_or_add_following_ids(461053984))
# print(DBManager.get_or_add_followers_ids(461053984))

def generate_follow_network():
    G = TweetUtils.TweetUtils().construct_follow_graph(None, 461053984, 100, False) # me
    print("Finished me\n")
    G.save("follow_graph.pickle")
    G = load("follow_graph.pickle")

    G = TweetUtils.TweetUtils().construct_follow_graph(None, 36858652, 100, False) # neill
    print("Finished Neill\n")
    G.save("follow_graph.pickle")
    G = load("follow_graph.pickle")

    G = TweetUtils.TweetUtils().construct_follow_graph(G, 67328299, 200, False) # earvin
    print("Finished Earvin\n")
    G.save("follow_graph.pickle")
    G = load("follow_graph.pickle")

    G = TweetUtils.TweetUtils().construct_follow_graph(None, 161196705, 1000, False) # clark
    print("Finished Clark\n")
    G.save("follow_graph.pickle")
    # G = load("follow_graph.pickle")

    labels = list(G.vs['screen_name'])
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

    color_list = ['#DD5E34', '#69CD45', '#6959CD', '#000005', '#FF00FF', '#FFFF00', '#808000', '#00FF00', '#00FFFF', '#008080', '#000080', '#800080']

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

    layout = Layout(title= "Follow Network",
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
    py.plot(fig, filename='github-network-community-igraph-{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))


generate_follow_network()
