class beam_profile(object):
    def __init__(self,amp,sigma1,sigma2,rho,mu1,mu2):
        self.amp     = amp
        self.sigma_1 = sigma1
        self.sigma_2 = sigma2
        self.rho     = rho
        self.mu1     = mu1
        self.mu2     = mu2

        self._p = self.amp/2./numpy.pi/self.sigma_1/self.sigma_2/ \
                numpy.sqrt(1-self.rho**2)

        self._a = 1./2./(1.-self.rho**2)

    def __call__(self,x1,x2):

        t1 = x1**2/self.sigma_1**2
        t2 = x2**2/self.sigma_2**2
        t12 = 2.*self.rho*x1*x2/self.sigma_1/self.sigma_2

        return self._p*numpy.exp(-self._a*(t1+t2-t12))


