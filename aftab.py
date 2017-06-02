import numpy as np

class lc:
	"""auxiliary class - just a container for lightcurve data"""
	def __init__(s, E, n, z, dL, thv, nu, t, F):
		
		s.nu_obs = nu
		s.thv = thv
		s.E = E
		s.n = n
		s.z = z
		s.dL = dL
		s.t = t
		s.F = F

class afterglow_table:
	"""This class allows to use a previously saved series of afterglow
	light curves as a basis for lightcurve generation, using interpolation
	and scaling relations."""
	
	def __init__(s, lc_list, tab_index, nu_list, thv_list, E0=1e50, n0=1., z0=0., dL0=1e28, thj=0.2):
		s.LC = lc_list
		s.idx = tab_index
		s.nu = nu_list
		s.thv = thv_list
		s.E0 = E0
		s.n0 = n0
		s.z0 = z0
		s.dL0 = dL0
		s.thj = thj
	
	def lightcurve(s, E, n, thv, nu, z, dL, no_points=60):
		"""Returns an afterglow lighcurve obtained from a table of
		BOXFIT v.1.0 (Van Eerten+12) lightcurves. 
		The parameters are:
		E = isotropic equivalent kinetic energy of the jet (in erg)
		n = number density of the interstellar medium (in cm^-3)
		thv = viewing angle (in radians)
		nu = observing frequency (in Hz)
		z = redshift
		dL = luminosity distance (in cm)
		no_points = the number of points at which to interpolate the lightcurve
		
		Returns a tuple (t,F), where F is the flux density in mJy and t is the time in days.
		
		CAVEATS:
		1) the method is only valid at low redshifts <<1 (no observer to rest frame 
		frequency shift taken into account).
		2) the observer frequency must be one of those in the original table. No interpolation
		on frequency is performed (will just use the nearest one available in the table).
		3) you can change E and n at your own risk: the lightcurves will be rescaled using 
		scaling relations. That should work fine for small displacements from the original E and n 
		used when creating the lightcurves, but there's no guarantee that the results are correct."""	
		
		j = np.argmin(np.abs(np.log(nu/s.nu)))
		i = int(np.floor(np.interp(thv,s.thv,np.arange(len(s.thv)))))
		if (s.thv[i]>thv and i>0) or i>len(s.thv)-2:
			i=i-1
		
		
		t = np.logspace(-1.,np.log10(512.),no_points)
		
		mu = (thv-s.thv[i])/(s.thv[i+1]-s.thv[i])
		
		if i<len(s.thv)-1:
			tA = (1.+z)/(1.+s.z0)*(E/s.E0/n*s.n0)**(1./3.)*s.LC[s.idx[i,j]].t
			tB = (1.+z)/(1.+s.z0)*(E/s.E0/n*s.n0)**(1./3.)*s.LC[s.idx[i+1,j]].t
			FA = (s.dL0/dL)**2*(E/s.E0*n/s.n0)*s.LC[s.idx[i,j]].F
			FB = (s.dL0/dL)**2*(E/s.E0*n/s.n0)*s.LC[s.idx[i+1,j]].F
			
			fa = np.interp(t,tA,FA)
			fb = np.interp(t,tB,FB)

			F = fa*(1.-mu) + fb*mu
			
		else:
			tA = (1.+z)/(1.+s.z0)*(E/s.E0/n*s.n0)**(1./3.)*s.LC[s.idx[i,j]].t
			FA = (s.dL0/dL)**2*(E/s.E0*n/s.n0)*s.LC[s.idx[i,j]].F

			F = np.interp(t,tA,FA)
		
		return t,F


def LoadTable(filename_root):
	"""Loads a table of previously computed afterglow light curves.
	The 'filename_root' is the initial part of the names of the files
	containing the table information."""
		
	tab_idx = np.load(filename_root + '_idx.npy')
	thv_list = np.load(filename_root + '_thv.npy')
	nu_list = np.load(filename_root + '_nu.npy')
	t = np.load(filename_root + '_t.npy')
	F_list = np.load(filename_root + '_F.npy')
	params = np.load(filename_root + '_params.npy')
	
	LC_list = []
	
	for i in range(len(thv_list)):
		for j in range(len(nu_list)):
			LC_list.append(lc(params[0],params[1],params[3],params[4],thv_list[i],nu_list[j],t,F_list[tab_idx[i,j]]))
	
	return afterglow_table(LC_list,tab_idx,nu_list,thv_list,E0=LC_list[0].E,n0=LC_list[0].n,z0=LC_list[0].z,dL0=LC_list[0].dL,thj=params[2])

def create_boxfit_table(outfilename, boxfitpath='/boxfit/bin', E=1e50, n=0.01, z=0., dL=1e28, thj=0.2, p=2.5, epsilonE=0.1, epsilonB=0.01, thv = np.linspace(0.,np.pi/2.,20), nu=np.array([1.5e8,5e9,3e10,1.84e14,4.56e14,2.4e17]), no_points = 60, t0 = 0.01, t1 = 512.):
	"""You can use this function to generate a table of lightcurves
	using Van Eerten's BOXFIT v1.0."""
	
	import BoxfitControl as bf
	
	bf.boxfit_path=boxfitpath
	
	#tell the user what you're going to do
	print(' A series of afterglow light curves is going to be generated\n' +\
	' using BOXFIT (Van Eerten et al. 2012).\n The parameters are the following:\n' +\
	'  E={} erg, n={} cm^-3, z={}, d_L={} cm,\n'.format(E,n,z,dL) +\
	'  theta_jet={} deg, epsilonE={}, epsilonB={}, p={}\n'.format(thj/np.pi*180.,epsilonE,epsilonB,p) +\
	' The viewing angles at which the light curves will be generated are (in deg):\n' +\
	'{}\n'.format(thv/np.pi*180.) +\
	' The observer frequencies are (in Hz):\n'
	'{}\n'.format(nu) +\
	' Each light curve will be expressed in mJy and will be\n computed in {} points between '.format(no_points) +\
	' t0={} days and t1={} days.\n The number of lightcurves to be generated is {}.\n Here we go...'.format(t0,t1,len(thv)*len(nu)))
	
	#------ table generation
	F_list = []
	Table_index = np.zeros([len(thv),len(nu)],dtype=int)
	idx = 0
	
	for i in range(len(thv)):
		for j in range(len(nu)):
			
			#the radial and angular cell resolution are optimized depending
			#on the viewing angle
			r_res = int(100 + 656*((np.pi/2.-thv[i])/np.pi*2.)**2.)
			phi_res = int(1 + 31*(thv[i]/np.pi*2.)**0.5)
			
			#a boxfitsettings file is created for each lightcurve
			bf.CreateBoxfitSettings('tab.{0}.{1}.txt'.format(i,j),E=E,n=n,theta_obs=thv[i],nu_0=nu[j],theta_0=thj,no_points=no_points,t_0=t0,t_1=t1,eds_phi_res=phi_res,eds_r_res=r_res,epsilon_E=epsilonE,epsilon_B=epsilonB, z=z, d_L = dL, p=p)
			#run boxfit and store the results
			t,F = bf.RunBoxfit('tab.{0}.{1}.txt'.format(i,j))
			F_list.append(F)
			Table_index[i,j]=idx
			idx = idx + 1
			print(idx)
			
	#---- the results are saved in a series of numpy files (not very
	# clean at the moment, there are some redundances...)
	np.save(outfilename + '_idx',Table_index)
	np.save(outfilename + '_nu',nu)
	np.save(outfilename + '_thv',thv)
	np.save(outfilename + '_t',t)
	np.save(outfilename + '_F',np.array(F_list))
	np.save(outfilename + '_params',np.array([E,n,thj,z,dL,epsilonE,epsilonB]))
	print('...done.')

	
	
