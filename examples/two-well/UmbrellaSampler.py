import os, sys
import numpy as np

class UmbrellaSampler(object):
    """A Sampler class that can perform biased sampling of the two-well potential."""
    
    def __init__(self,
                 x0_values = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                 kspring_values  = ([0.0]+[2.0]*9) ):
        """Initialize the UmbrellaSampler class.
        
        PARAMETERS
        x0_values - a list of x0 positions of the umbrella anchors
        kspring_values  - a list of force constant values, in kcal/mol/[x]^2
        """
        
        self.x0_values = x0_values  # Umbrella anchor positions
        self.kspring_values = kspring_values    # force constant values, in kcal/mol/[x]^2
        
        assert len(self.x0_values) == len(self.kspring_values)
        
        self.K = len(self.x0_values)  # the number of thermodynamic ensembles
        
    
    def U(self, x, x0=0.0, kspring=0.0):
        """Return the value of the **biased** 1D potential energy surface in kcal/mol:

        U(x) = -2 \ln [ e^{-2(x-2)^2-2} + e^{-2(x-5)^2} ] + (k/2.)(x - x0)**2
        """
        result = -2.0*np.log( np.exp(-2.0*(x-2)**2 - 2) + np.exp(-2.0*(x-5)**2) )
        result += kspring/2.*(x - x0)**2
        return result

    
    def dU_kl(self, x, k, l):
        """Return the difference of the potential energy $\Delta U_kl(x)$ for a snapshot
        from ensemble index k, re-evaluated in ensemble index l:
        
        \Delta U_kl(x) = (bias from ens. l) - (bias from ens k.)
        
        """
        result = self.kspring_values[l]/2.*(x - self.x0_values[l])**2
        result -= self.kspring_values[k]/2.*(x - self.x0_values[k])**2
         
        return result


    def sample(self, xinit, nsteps, thermo_index, djump=0.05, xmin=1.5, xmax=5.5,
               kT=0.596, nstride=100, nprint=10000, verbose=False):
        """Perform Monte Carlo sampling of the potential energy surface U
        by 

        INPUT
        xinit        - the starting position
        nsteps       - number of steps of Monte Carlo to perform
        thermo_index - the index of the thermodynamic ensemble to sample
        
        PARAMS
        djump    attempt random moves drawn from [-djump, +djump]
        xmin     reject moves x < xmin
        xmax     reject moves x > xmax
        kT       thermal energy in units of kcal/mol (Default: 0.596)
        nstride  frequency of step to subsample the trajectory

        Note:  the djump=0.005 parameter is from the 2017 Stelzl et al. paper    
        """
        
        assert (thermo_index < self.K)
        
        x = xinit
        energy = self.U(x, self.x0_values[thermo_index], self.kspring_values[thermo_index])

        step = 0
        accepted_steps = 0
        traj = np.zeros( int(nsteps/nstride) )
        itraj = 0

        # pre-calculate random numbers
        r = np.random.random( nsteps )
        s = np.random.random( nsteps )

        while step < nsteps:

            xnew = x + djump*(2.0*s[step]-1.0)
            new_energy = self.U(xnew, self.x0_values[thermo_index], self.kspring_values[thermo_index])

            # calculate Metropolis acceptance 
            accept = (r[step] < min(1, np.exp( -1.0*(new_energy-energy)/kT ) ))

            # reject moves that bring x outside the range
            accept = accept*(xnew>xmin)*(xnew<xmax)

            if accept:
                accepted_steps += 1
                x = xnew
                energy = self.U(x, self.x0_values[thermo_index], self.kspring_values[thermo_index])

            if step%nstride == 0:
                traj[itraj] = x
                itraj += 1

            if verbose:
                if step%nprint == 0:
                    print('step', step, 'of', nsteps, ': x =', x, 'energy =', energy)

            step += 1
            acc_ratio = float(accepted_steps)/float(step)

        return traj


