import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import pandas as pd


class GaussDistribution(ContinuousDistribution):
    def __init__(self, dim=1):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.covariance = np.eye(dim)
        self.data = pd.DataFrame()
        self.samples = None
    
    def import_data(self, file_path):
        # implementation to import data from file
        self.data = pd.read_csv(file_path)
    
    def export_data(self, data, file_path):
        # implementation to export data to file
        df = pd.DataFrame(data)
        df.to_csv(file_path)
    
    def compute_mean(self, data):
        self.mean = np.mean(data, axis=0)
    
    def compute_standard_deviation(self, data):
        self.covariance = np.cov(data, rowvar=False)
    
    def visualize(self, data=None):
        if data is None:
            data = multivariate_normal.rvs(mean=self.mean, cov=self.covariance, size=1000)
        
        if self.dim == 1:
            mean = 0
            covariance = 0.8
            x = np.linspace(mean - 3*np.sqrt(covariance), mean + 3*np.sqrt(covariance), 100)
            plt.plot(x, multivariate_normal.pdf(x, mean=mean, cov=covariance), color = 'blue')
            plt.title(f'1D Gaussian Distribution with a mean of {mean} and a covariance of {covariance}')
            
            plt.savefig('gaussian1D.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            
        elif self.dim == 2:
            covariance = np.array([[1, 0.8],
                                     [0.8, 1]])
            mean = np.array([0, 0])
            x, y = np.mgrid[mean[0]-3*np.sqrt(covariance[0,0]):mean[0]+3*np.sqrt(covariance[0,0]):.01,
                            mean[1]-3*np.sqrt(covariance[1,1]):mean[1]+3*np.sqrt(covariance[1,1]):.01]
            pos = np.empty(x.shape + (2,))
            pos[:, :, 0] = x
            pos[:, :, 1] = y
            rv = multivariate_normal(mean, covariance)

            # Generating the density function
            # for each point in the meshgrid
            pdf = np.zeros(x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    pdf[i,j] = rv.pdf([x[i,j], y[i,j]])
            
            pdf_list = []
            fig = plt.figure()

            # Plotting the density function values
            bx = fig.add_subplot(131, projection = '3d')
            bx.plot_surface(x, y, pdf, cmap = 'viridis')
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title(f'2D Gaussian Distribution with a mean of {mean} and a covariance of {covariance}')
            pdf_list.append(pdf)
            bx.axes.zaxis.set_ticks([])
            
            plt.tight_layout()
            
            plt.savefig('gaussian2D_surface.pdf', bbox_inches='tight', transparent=True)
            plt.show()

            # Plotting contour plots
            for idx, val in enumerate(pdf_list):
                plt.subplot(1,3,idx+1)
                plt.contourf(x, y, val, cmap='viridis')
                plt.xlabel("x1")
                plt.ylabel("x2")
            plt.tight_layout()
            plt.title(f'2D Gaussian Distribution Contour with a mean of {mean} and a covariance of {covariance}')
            
            plt.savefig('gaussian2D_contour.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            
        elif self.dim == 3:
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            x, y, z = np.mgrid[self.mean[0]-3*np.sqrt(self.covariance[0,0]):self.mean[0]+3*np.sqrt(self.covariance[0,0]):.1,
                               self.mean[1]-3*np.sqrt(self.covariance[1,1]):self.mean[1]+3*np.sqrt(self.covariance[1,1]):.1,
                               self.mean[2]-3*np.sqrt(self.covariance[2,2]):self.mean[2]+3*np.sqrt(self.covariance[2,2]):.1]
            #Plot the samples from the file

            ax1.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c='blue', alpha=0.5)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('Data from File')

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.scatter(self.samples[:, 0], self.samples[:, 1], self.samples[:, 2], c='orange', alpha=0.5)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.set_title('Data from Samples')

            plt.savefig('gaussian3D.pdf', bbox_inches='tight', transparent=True)
            plt.show()
            
    def generate_samples(self, n_samples):
        self.samples = multivariate_normal.rvs(mean=self.mean, cov=self.covariance, size=n_samples)