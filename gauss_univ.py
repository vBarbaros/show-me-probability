#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

from scipy.stats import norm
from scipy.integrate import quad

mu1 = 16
mu2 = 0
sigma1 = 1
sigma2 = 1
img_nr = 2
def p(x):
    return norm.pdf(x, mu1, sigma1)

def q(x):
	#norm.pdf(x, loc==mean, scale==sigma)
    return norm.pdf(x, mu2, sigma2)

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

def symKL(x):
	return ( (p(x) * np.log( p(x) / q(x)) ) + ( q(x) * np.log( q(x) / p(x) )) )

range = np.arange(-30, 30, 0.001)

KL_int, err = quad(KL, -10, 10)
symKL_int, err = quad(symKL, -10, 10) 
print 'KL: ', KL_int
print 'symKL: ', symKL_int 
fig = plt.figure(figsize=(10, 10), dpi=100)

def plot_KL():

	KL_int, err = quad(KL, -10, 10)
	print 'KL: ', KL_int
	#---------- First Plot

	ax = fig.add_subplot(1,2,1)
	ax.grid(True)
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	ax.set_xlim(-10,10)
	ax.set_ylim(-0.1,0.25)

	ax.text(-4.5, 0.17, 'p(x) \n\n$\mu$='+str(mu1)+'\n$\sigma$='+str(sigma1), horizontalalignment='center',fontsize=12,color='b')
	ax.text(4.5, 0.17, 'q(x) \n\n$\mu$='+str(mu2)+'\n$\sigma$='+str(sigma2), horizontalalignment='center',fontsize=12,color='g')

	plt.plot(range, p(range))
	plt.plot(range, q(range))

	#---------- Second Plot
	
	ax = fig.add_subplot(1,2,2)
	ax.grid(True)
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	ax.set_xlim(-10,10)
	ax.set_ylim(-0.1,0.25)

	ax.text(-4.0, -0.07, '$KL_{Div}(p||q)$\n$KL_{Div}$='+str(KL_int), horizontalalignment='center',fontsize=12,color='b')

	ax.plot(range, KL(range))

	ax.fill_between(range, 0, KL(range))
	plt.savefig('KL'+str(img_nr)+'.png',bbox_inches='tight')
	plt.show()	

def plot_sym_avg_KL():


	sym_avg_KL_int, err = quad(symKL, -20, 20) 
	print 'symKL: ', sym_avg_KL_int 
	#First Plot

	ax = fig.add_subplot(1,2,1)
	ax.grid(True)
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	ax.set_xlim(-35,35)
	ax.set_ylim(-0.1,0.25)

	ax.text(-7.5, -0.04, 'p(x) \n\n$\mu$='+str(mu1)+'\n$\sigma$='+str(sigma1), horizontalalignment='center',fontsize=12,color='b')
	ax.text(7.5, -0.04, 'q(x) \n\n$\mu$='+str(mu2)+'\n$\sigma$='+str(sigma2), horizontalalignment='center',fontsize=12,color='g')

	plt.plot(range, p(range))
	plt.plot(range, q(range))

	#Third Plot
	ax = fig.add_subplot(1,2,2)
	ax.grid(True)
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	ax.set_xlim(-35,35)
	ax.set_ylim(-0.1,0.25)

	ax.text(-7.5, -0.04, '$sym_KL_{Div}{(p||q)+(q||p)}$\n$sym_KL_{Div}$='+str(sym_avg_KL_int), horizontalalignment='center',fontsize=12,color='b')

	ax.plot(range, symKL(range))

	ax.fill_between(range, 0, symKL(range))

	plt.savefig('1_SymKL'+str(img_nr)+'.png',bbox_inches='tight')
	plt.show()

"""

def KL_formula(mu1, sigma1, mu2, sigma2):
	#print "(np.log(sigma2 / sigma1)) ", (np.log(sigma2 / sigma1))
	return ( (np.log(sigma2 / sigma1)) + (( pow(sigma1,2) + pow((mu1-mu2),2) ) / (2.0 * pow(sigma2, 2))) - 0.5) 



#plot_KL()

#plot_sym_avg_KL()



def sym_KL_div(mu1, sigma1, mu2, sigma2):
	#print 'KL_formula(mu1, sigma1, mu2, sigma2)', KL_formula(mu1, sigma1, mu2, sigma2)
	#print 'KL_formula(mu2, sigma2, mu1, sigma1)', KL_formula(mu2, sigma2, mu1, sigma1)
	#return KL_formula(mu1, sigma1, mu2, sigma2) + KL_formula(mu2, sigma2, mu1, sigma1)
	#print sigma1, sigma2 , (np.log(sigma2 / sigma1))
	return ( np.log(sigma2 / sigma1) + (( pow(sigma1,2) + pow((mu1-mu2),2) ) / (2.0 * pow(sigma2, 2))) - 0.5) + ( np.log(sigma1 / sigma2) + (( pow(sigma2,2) + pow((mu2-mu1),2) ) / (2.0 * pow(sigma1, 2))) - 0.5)

def sym_KL_div_avg(mu1, sigma1, mu2, sigma2):
	return ( ( pow((mu1 - mu2), 2)*( pow(sigma1,2)+pow(sigma2,2)) + pow(sigma1, 4)+ pow(sigma2, 4) )/( 2*pow(sigma1,2)*pow(sigma2,2)) - 1)

mu_lst = [0, 1.0/32.0 , 1.0/16.0 , 1.0/8.0, 1.0/4.0, 1.0/2.0, 1, 2, 4, 8, 16, 32]

mu1_1, sigma2_2, sigma1_1 = 0, 1, 1


#p = np.random.normal(mu, sigma, 1000)
#q = np.random.normal(mu, sigma, 1000)

def p(x):
    return norm.pdf(x, mu_p, sigma_p)

def q(x):
	#norm.pdf(x, loc==mean, scale==sigma)
    return norm.pdf(x, mu_q, sigma_q)

def KL(x):
    return p(x) * np.log( p(x) / q(x) )

def symKL(x):
	return ( (p(x) * np.log( p(x) / q(x)) ) + ( q(x) * np.log( q(x) / p(x) )) )


for i in mu_lst:
	mu_p, sigma_p = 0, 1 # mean and standard deviation
	mu_q, sigma_q = i, 1
	KL_int, err = quad(KL, -15, 15)
	symKL_int, err = quad(symKL, -15, 15) 
	print 'sigma_q: ', i, 'KL: ', KL_int
	print 'sigma_q: ', i, 'symKL: ', symKL_int
	print 'sigma_q: ', i, 'sym_avgKL: ', symKL_int*0.5 

#for i in mu_lst:
	#print "sigma2_2 = ", i, "KL = ", KL_formula(mu1_1, sigma1_1, i, sigma2_2)
	#print "mu2_2 = ", i, "symKL = ", sym_KL_div(mu1_1, sigma1_1, i, sigma2_2)
	#print "sigma1_1 = ", i, "symKL_avg = ", sym_KL_div_avg(mu1_1, i, mu2_2, sigma2_2)
"""
