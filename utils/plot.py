import visdom
import numpy as np


class Plot(object):
    def __init__(self):
        self.viz = visdom.Visdom()

    def vis_line(self, yVal, xVal, label, title, win=None):
        if win:
            return self.viz.line(X=np.array([xVal]),
                                 Y=np.array([yVal]),
                                 win=win,
                                 name=label,
                                 update='append',
                                 opts=dict(title=title,
                                           showlegend=True)
                                 )
        else:
            return self.viz.line(X=np.array([xVal]),
                                 Y=np.array([yVal]),
                                 name=label,
                                 opts=dict(title=title,
                                           showlegend=True)
                                 )

    def vis_heatmap(self, mat, title, win=None):
        return self.viz.heatmap(
            X=np.flipud(mat),
            win=win,
            opts=dict(
                columnnames=['False', 'True'],
                rownames=['True', 'False'],
                title=title
            ))

    def vis_image_grids(self, grid, title, win=None):
        nrows = min(int(grid.shape[0]/10), 4)
        return self.viz.images(
            grid, nrow=nrows, padding=2, win=win, opts=dict(title=title))
