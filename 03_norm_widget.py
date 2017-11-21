import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as sps
from scipy.stats import beta
from matplotlib.widgets import Button

class Index(object):
	def __init__(self):
		self.plt = plt.subplots()
		self.fig, self.ax = self.plt
		self.fig.dpi = 85
		self.fig.subplots_adjust(bottom=0.2)
		self.sigma_next = 0.1
		self.step_incr_decr = 0.1
		self.sigma_nod = 0.1
		self.mu, self.sigma = 0.5, self.sigma_nod

		self.theta1, self.theta2 = 0, 2
 
		self.shape, self.scale = 2, 2

		self.alpha, self.beta = 2.0, 0.5

		self.bins = 75
		self.x_coord = -4.5, 4.5
		self.y_coord = 0, 5
		self.ax.set_xlim(self.x_coord)
		self.ax.set_ylim(self.y_coord)
		self.x_text, self.y_text = -0.7*(self.x_coord[1]), 0.6*(self.x_coord[1])
		self.ax.text(self.x_text, self.y_text, '\n\n$\mu$='+str(self.mu)+'\n$\sigma$='+str(self.sigma), horizontalalignment='center',fontsize=12,color='b')
		#self.pdf = self.normal_pdf()
		self.pdf = self.beta_pdf()
		self.chard = "N"
		callback = self

		axunif = plt.axes([0.45, 0.05, 0.13, 0.075])
		bunif = Button(axunif, 'Uniform')
		bunif.on_clicked(callback.uniform_pdf_ch)

		axgamma = plt.axes([0.30, 0.05, 0.13, 0.075])
		bgamma = Button(axgamma, 'Gamma')
		bgamma.on_clicked(callback.gamma_pdf_ch)

		axbeta = plt.axes([0.15, 0.05, 0.13, 0.075])
		bgbeta = Button(axbeta, 'Beta')
		bgbeta.on_clicked(callback.beta_pdf_ch)

		axprev = plt.axes([0.61, 0.05, 0.13, 0.075])
		axnext = plt.axes([0.77, 0.05, 0.13, 0.075])
		bnext = Button(axnext, 'Increase $\sigma$')
		bnext.on_clicked(callback.next)
		bprev = Button(axprev, 'Decrease $\sigma$')
		bprev.on_clicked(callback.prev)
		plt.show()

	def next(self, event):
		self.ax.cla()
		self.sigma_next+=self.step_incr_decr
		self.sigma = self.sigma_next 
		if self.chard == "N": # keep self.mu constant if Normal distribution
			self.pdf = self.normal_pdf()
		elif self.chard == "U":
			self.mu = (self.theta1 + self.theta2) / 2
			self.pdf = self.uniform_pdf()
		elif self.chard == "G":
			self.mu = self.scale * self.shape
			self.pdf = self.gamma_pdf()
		elif self.chard == "B":
			self.beta +=0.1
			self.pdf = self.beta_pdf()
		"""
		s = np.random.normal(self.mu, self.sigma, 1000)
		count, bins, ignored = self.ax.hist(s, self.bins, normed=True)
		x = 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.mu)**2 / (2 * self.sigma**2) )
		l, = self.ax.plot(bins,x,linewidth=3, color='r')
		"""
		self.ax.set_xlim(self.x_coord)
		self.ax.set_ylim(self.y_coord)
		self.ax.text(self.x_text, self.y_text, '\n\n$\mu$='+str(self.mu)+'\n$\sigma$='+str(self.sigma), horizontalalignment='center',fontsize=12,color='b')
		plt.draw()

	def prev(self, event):
		self.ax.cla()
		self.sigma_next-=self.step_incr_decr
		self.sigma = self.sigma_next 
		if self.chard == "N": # keep self.mu constant if Normal distribution
			self.pdf = self.normal_pdf()
		elif self.chard == "U":
			self.mu = (self.theta1 + self.theta2) / 2
			self.pdf = self.uniform_pdf()
		elif self.chard == "G":
			self.mu = self.scale * self.shape
			self.pdf = self.gamma_pdf()
		elif self.chard == "B":
			self.beta -=0.1
			self.pdf = self.beta_pdf()

		self.ax.set_xlim(self.x_coord)
		self.ax.set_ylim(self.y_coord)
		self.ax.text(self.x_text, self.y_text, '\n\n$\mu$='+str(self.mu)+'\n$\sigma$='+str(self.sigma), horizontalalignment='center',fontsize=12,color='b')
		plt.draw()

	def normal_pdf(self):
		s = np.random.normal(self.mu, self.sigma, 1000)
		count, bins, ignored = self.ax.hist(s, self.bins, normed=True)
		x = 1/(self.sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.mu)**2 / (2 * self.sigma**2) )
		l, = self.ax.plot(bins,x,linewidth=3, color='r')

	def uniform_pdf(self):
		self.theta2 = self.sigma * math.sqrt(12) + self.theta1
		#self.mu = (self.theta1 + self.theta2) / 2
		s = np.random.uniform(self.theta1, self.theta2, 1000)
		count, bins, ignored = self.ax.hist(s, self.bins, normed=True)
		x = 1 / (self.theta2 - self.theta1)
		a = np.ones_like(bins)
		a.fill(x)
		#l, = self.ax.plot(bins,np.ones_like(bins),linewidth=3, color='r')
		l, = self.ax.plot(bins,a,linewidth=3, color='r')

	def uniform_pdf_ch(self, event):
		self.chard = "U"

	def gamma_pdf(self):
		self.scale = math.sqrt(self.shape)/self.sigma
		#self.shape = self.scale / self.mu
		#self.shape = self.scale / self.mu
		#self.mu = self.scale * self.shape
		s = np.random.gamma(self.shape, self.scale, 1000)
		count, bins, ignored = self.ax.hist(s, self.bins, normed=True)
		x = bins**(self.shape-1)*(np.exp(-bins/self.scale) / (sps.gamma(self.shape)*self.scale**self.shape))
		l, = self.ax.plot(bins,x,linewidth=3, color='r')

	def gamma_pdf_ch(self, event):
		self.chard = "G"

	def beta_pdf(self):
		#a, b = 2.31, 0.627
		s = np.random.beta(self.alpha, self.beta, size=1000)
		count, bins, ignored = self.ax.hist(s, normed=True)
		x = np.linspace(beta.ppf(0.01, self.alpha, self.beta),beta.ppf(0.99, self.alpha, self.beta), 100)
		l, = self.ax.plot(x, beta.pdf(x, self.alpha, self.beta),'r-', lw=3, alpha=0.6, label='beta pdf')

	def beta_pdf_ch(self, event):
		self.chard = "B"
if __name__ == '__main__':
	ind = Index()
	
