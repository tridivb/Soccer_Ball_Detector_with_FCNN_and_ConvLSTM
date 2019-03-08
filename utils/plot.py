import visdom
import numpy as np


class Plot(object):
    def __init__(self):
        self.viz = visdom.Visdom()

    def vis_line(self, yVal, xVal, label, title, win=None):
        """ Plot 2D graphs for multiple data points
        Args:
            yVal (double/int): Y-axis value to plot
            xVal (double/int): X-axis value to plot
            label (String): Label of current plot
            title (String): Title of plot window
            win (visdom.line object): Window to use for updating(optional) [Default: None]
        
        Returns:
            win (visdom.line object): Visdom.line object to reuse
        """
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
        """ Plot 2D heatmaps
        Args:
            mat (2D numpy array): Array to plot heatmap for
            title (String): Title of plot window
            win (visdom.heatmap object): Window to use for updating(optional) [Default: None]
        
        Returns:
            win (visdom.heatmap object): Visdom.heatmap object to reuse
        """
        return self.viz.heatmap(
            X=np.flipud(mat),
            win=win,
            opts=dict(
                columnnames=['False', 'True'],
                rownames=['True', 'False'],
                title=title
            ))

    def vis_image_grids(self, grid, title, nrows=3, win=None):
        """ Plot 2D image grids
        Args:
            grid (Tensor): Tensor to plot image grid
            title (String): Title of plot window
            nrows (int): Number of images to plot in a row
            win (visdom.images object): Window to use for updating(optional) [Default: None]
        
        Returns:
            win (visdom.images object): Visdom.images object to reuse
        """
        # nrows = 3
        return self.viz.images(
            grid, nrow=nrows, padding=3, win=win, opts=dict(title=title))
