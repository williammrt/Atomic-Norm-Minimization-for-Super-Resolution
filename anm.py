"""
You must install numpy, scipy, cvxpy, pandas and MOSEK(or other SDP solver supported by cvxpy, which is CVXOPT, the default solver SCS comes with cvxpy 
can not solve accurately) to use this program
matplotlib is needed only when you want to see the graph of psf and signal and solution found
See the how to request MOSEK academic license at https://www.mosek.com/products/academic-licenses/
"""
import numpy as np
import scipy as sp
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import time
start = time.time()

PLOT_ANS = False
BANDWIDTH = 20
NOISE = 0.05
MAX_freq = int(BANDWIDTH/2)
n = 2 * int(BANDWIDTH/2) + 1

def test(NUM_POINT, DIST):
	#print(NUM_POINT, DIST)
	pos = np.linspace(0.1, 0.1+NUM_POINT*DIST, NUM_POINT, dtype=complex, endpoint=False)

	#pos = np.array([0.1, 0.2, 0.25,  0.3, 0.85], dtype=complex) # [0,1)
	amp = np.array([i*(-1)**i + i*1j for i in range(1,NUM_POINT+1)], dtype=complex)

	# point spread function
	def psf(x): 
		return np.sum([np.cos(2.0*np.pi*x*i) for i in range(1,MAX_freq+1)],axis=0)+0.5
		
	# signal we received
	def signal(x):
		assert(len(pos) == len(amp))
		p_sz = len(pos)
		result = np.zeros(x.size, dtype=complex)
		for i in range(0,p_sz):
			result += amp[i]*psf(x-pos[i])
		return result + (1+1j)*np.random.normal(0,NOISE,size=x.size)

	def get_freq(N,T, func):
		x = np.linspace(0.0, N*T, N, endpoint=False)
		y = func(x)
		yf = sp.fft.fft(y)/N
		yfreqs = np.concatenate((yf[-MAX_freq:], yf[:MAX_freq+1]))
		return yfreqs

	def print_array(name, arr):
		df = pd.DataFrame({name:arr})
		df.index += 1
		print(df)
		print()

	def solve_convex(x):
		delta = np.zeros(n)
		delta[0] = 1
		cst = np.reshape(1,(1,1))
		A = np.zeros((n,n,n))

		for i in range(n):
			A[:,:,i] = np.eye(n, k=i)

		p = cp.Variable((n,1), complex=True)
		H = cp.Variable((n,n), hermitian=True)
		c = cp.Constant(cst)
		constraints = [ cp.vstack([cp.hstack([H,p]), cp.hstack([p.H, c])])>> 0]
		constraints += [cp.trace(A[:,:,i]@H) == delta[i] for i in range(n)]
		prob = cp.Problem(cp.Maximize(cp.real(cp.scalar_product(x,p))), constraints)
		ans = prob.solve(solver=cp.MOSEK)
		return p.value.reshape(n)

	def get_peaks(p):
		ptau = np.polynomial.Polynomial(p)
		N = 100000
		x_plt = np.linspace(0,1, N, endpoint = False)
		points = np.exp(1.0j*2*np.pi*x_plt)
		y_plt = np.abs(ptau(points))

		# find pos
		peaks, _ = sp.signal.find_peaks(y_plt, height=0.99999)
		peaks = peaks/ N 
		return peaks
		"""
		# find amp by solving linear equations
		r = len(peaks)
		rx = np.linspace(0,1,r,endpoint = False, dtype=np.longdouble)
		A = np.hstack([psf(rx-pos).reshape(1,r).transpose() for pos in peaks])

		b = np.transpose(signal(rx))
		amps = np.linalg.solve(A,b)
		#amps = mp.lu_solve(A,b)
		"""

	def get_amps(peaks):
		A = np.vstack([ np.exp(-1j* np.pi * 2 * f * peaks) for f in freqs])
		b = np.transpose(yfreqs)
		#amps = np.real(sp.linalg.lstsq(A,b)[0])
		amps = sp.linalg.lstsq(A,b)[0]
		return amps*2

	# sample frequency, we can find at most N/2 - 1  freq
	N = 100
	# sampling space
	T = 1 / N 

	f_tmp = sp.fft.fftfreq(N, T)
	freqs = np.concatenate((f_tmp[-MAX_freq:], f_tmp[:MAX_freq+1]))

	yfreqs = get_freq(N, T, signal)
	gfreqs = get_freq(N, T, psf)

	x = yfreqs / gfreqs

	CO_start = time.time()
	p = solve_convex(x)
	CO_end = time.time()

	peaks = get_peaks(p)

	amps = get_amps(peaks)

	if (len(peaks) == len(pos)):
		pn = np.linalg.norm(peaks - pos)
		rpn = pn/np.linalg.norm(pos)
		#print(f"pos norm= {pn}, relative diff= {pn/np.linalg.norm(pos)}")
		#return round(rpn,6)
		return rpn
		an = np.linalg.norm(amps - amp)
		print(f"amp norm= {an}, relative diff= {an/np.linalg.norm(amp)}")
	#return -1
	# output peaks and amps as table
	
	df = pd.DataFrame({"peaks": peaks, "amps": amps})
	df.index += 1
	print(df)
	df = pd.DataFrame({"pos":pos, "amp":amp})
	df.index+= 1
	print(df)
	return -1
	if PLOT_ANS: 
		sample_x = np.linspace(0.00001,1,1000, endpoint = False)

		#plot graph
		plt.plot(sample_x, signal(sample_x), label='signal(x)')
		plt.plot(sample_x, psf(sample_x), label='psf(x)')

		#plot solutions
		plt.plot(pos, amp, 'o', label="True ans")
		plt.plot(peaks, amps, 'x', label="Result")

		plt.xlim(0,1)
		plt.legend(loc='best')
		plt.grid()
		plt.show()

	end = time.time()
	print(f"totol time used = {end - start} seconds")
	print(f"Convex Opt time used = {CO_end - CO_start} seconds")

ans = [[]]
for i in range(2,10+1):
	ans += [[test(i,j*0.01) for j in range(1,10+1)]]

df = pd.DataFrame(ans[1:], columns =[0.01*i for i in range(1,10+1)],
                                           dtype = float) 
df = df.round(5)
df.index += 2
print(df)
