from matplotlib import colormaps
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02, odd_range=1):
    '''
    делает координатную сетку по входным точкам
    
    x, y: исходные точки, "вокруг которых" делаем сетку
    h: шаг сетки
    odd_range: величина zoom-out'a от точек х, y
    '''
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    x_min, x_max = x.min() - odd_range * x_range, x.max() + odd_range * x_range
    y_min, y_max = y.min() - odd_range * y_range, y.max() + odd_range * y_range
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
    '''
    предсказываем моделью точки сетки и отрисовываем разделяющую прямую и цвета
    
    xx, yy: точки сетки
    '''
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, **params, alpha=0.5, levels=np.unique(Z))

    
def plot_classification(model, x, y, hue, odd_range=0.3, mesh_h=0.02, figsize=None):
    '''
    рисует разделяющую прямую модели для 2D-задачи классификации
    
    model: что-то, у чего есть метод .predict, который дает ту картину предсказаний, какую вы хотите
    x, y: точки выборки, которые нужно отрисовать вместе с разделяющей прямой
    hue: их классы
    
    odd_range: чем больше, тем больше zoom-out от поданых на вход точек
    mesh_h: шаг сетки
    '''
    # сине-красная палитра
    cmap = colormaps['tab10']
    palette = sns.mpl_palette('tab10', n_colors=np.unique(hue).shape[0])
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # делаем частую сетку из точек "с центром" в поданных точках x, y
    xx, yy = make_meshgrid(x, y, h=mesh_h, odd_range=odd_range)
    
    # предсказываем моделью все точки сетки, отрисовываем цвета
    plot_contours(ax, model, xx, yy, cmap=cmap)
    
    # отрисовываем туда же поданные на вход точки x, y
    sns.scatterplot(x=x, y=y, hue=hue, hue_order=np.unique(hue), palette=palette, marker='o',
                    edgecolor='black', ax=ax, legend=True)