import numpy as np
import matplotlib.pyplot as plt
import aftab

#load a table
my_aftab = aftab.LoadTable('data/eb0.01thj0.2E50n0.01')

#print out the parameters of the original lightcurves
print(" The lightcurves in the table were created with the following parameters:\n")
print("E = {} erg".format(my_aftab.E0))
print("n = {} cm^-3".format(my_aftab.n0))
print("z = {}".format(my_aftab.z0))
print("dL = {} cm".format(my_aftab.dL0))
print("theta_jet = {} rad".format(my_aftab.thj))

print(" The available frequencies are:\n")
print("{} Hz".format(my_aftab.nu))

print(" The lightcurves are interpolated from the following viewing angles:\n")
print("{} rad".format(my_aftab.thv))


#generate some lightcurves and plot them
E = 1e50
n = 0.01
z = 0.
dL = 1e28

#viewing angles
thv = np.linspace(0.,1.,5)
#frequencies
nu = [5e9,4.56e14,2.4e17]
nu_labels = ['Radio','Optical','X-ray']
nu_colors = ['green','red','blue']

for i in range(len(nu)):
			
	fig = plt.figure(nu_labels[i])
	ax = fig.add_subplot(111)
	
	for tv in thv:
		t,F = my_aftab.lightcurve(E,n,tv,nu[i],z,dL)
		ax.plot(t,F,lw=3,c=nu_colors[i])
	
	ax.set_xlabel('t [days]')
	ax.set_ylabel('Flux density [mJy]')
	ax.loglog()

plt.show()
