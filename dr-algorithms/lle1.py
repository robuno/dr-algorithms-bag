import pandas as pd 
import numpy as np 
import plotly.express as px 

from sklearn.datasets import make_swiss_roll 
from sklearn.manifold import LocallyLinearEmbedding as LLE 
from sklearn.manifold import Isomap 

# Create a swiss roll
X, y = make_swiss_roll(n_samples=1000, noise=0.05)
X[:, 1] *= .8
X_x=np.zeros((300,1))
X_y=np.random.uniform(low=0, high=8, size=(300,1))
X_z=np.random.uniform(low=14, high=24, size=(300,1))
X2=np.concatenate((X_x, X_y, X_z), axis=1)
y2=X_z.reshape(300)

def Plot3D(X, y, plot_name):
    fig = px.scatter_3d(None, 
                        x=X[:,0], y=X[:,1], z=X[:,2],
                        color=y,
                        height=800, width=800
                       )
    
    fig.update_layout(title_text=plot_name,
                      showlegend=False,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))
    
    fig.update_traces(marker=dict(size=5))
    fig.update(layout_coloraxis_showscale=False)
    return fig


def Plot2D(X, y, plot_name):
    fig = px.scatter(None, x=X[:,0], y=X[:,1], 
                     labels={
                         "x": "Dimension 1",
                         "y": "Dimension 2",
                     },
                     opacity=1, color=y)

    fig.update_layout(dict(plot_bgcolor = 'white'))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                     zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                     showline=True, linewidth=1, linecolor='black')

    fig.update_layout(title_text=plot_name)
    fig.update_traces(marker=dict(size=7))
    return fig


firstSwiss = Plot3D(X, y, "Swiss Roll Example 1")
#firstSwiss.show()

#firstSwiss = Plot2D(X, y, ""Swiss Roll in 2D Form")
#firstSwiss.show()


s_lle_obj = LLE(n_neighbors=30, # no of neighbors
                    n_components=2, # new dimension
                    method="standard", # standard, hessian, modified, ltsa
                   )
    
std_lle_res = s_lle_obj.fit_transform(X)

sLLE_2D_fig = Plot2D(std_lle_res, y, 'Standard LLE Swiss Roll Example on 2D with 2 Dimensions with N=30')
sLLE_2D_fig.show()



m_lle_obj = LLE(n_neighbors=30, # no of neighbors
                    n_components=2, # new dimension
                    method="modified", # standard, hessian, modified, ltsa
                   )
    
modified_lle_res = m_lle_obj.fit_transform(X)

#mLLE_2D_fig = Plot2D(modified_lle_res, y, 'Modified LLE Swiss Roll Example on 2D with 2 Dimensions with N=30')
#mLLE_2D_fig.show()