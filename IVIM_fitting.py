import nibabel as nb
import sys
import csv
import numpy as np
import scipy.optimize as op
import scipy.stats as st
import getopt as go
import fitting_diffusion as ic
import multiprocessing as mp


__author__ = 'eric'


def ricepdf(x, v, s):
    # evaluates the rician probability density function of v, s at x
    return st.rice.pdf(x, v/s, scale=s)


def IVIM_fun_wrap(x, b):
    return IVIM_fun(b, x[0], x[1], x[2], x[3])


def IVIM_grad_wrap(x, b):
    return IVIM_grad(b, x[0], x[1], x[2], x[3])


def IVIM_grad_lsqr(x, b, s):
    return IVIM_grad_wrap(x, b)


def IVIM_grad_lsqr_sumsq(x, b, s):
    # this function evaluates the jacobian relative to S0, f, D, D* at the values b on the diffusion curve
    dS0 = np.sum(IVIM_dS0(b, x[0], x[1], x[2], x[3]))
    df = np.sum(IVIM_df(b, x[0], x[1], x[2], x[3]))
    dD = np.sum(IVIM_dD(b, x[0], x[1], x[2], x[3]))
    dDstar = np.sum(IVIM_dDstar(b, x[0], x[1], x[2], x[3]))
    return [dS0, df, dD, dDstar]


def IVIM_grad_lsqr_S0_D(x, b, s, x_other):
    dS0 = IVIM_dS0(b, x[0], x_other[0], x[1], x_other[1])
    dD = IVIM_dD(b, x[0], x_other[0], x[1], x_other[1])
    return [dS0, dD]


def IVIM_grad_lsqr_S0_D_sumsq(x, b, s, x_other):
    # why are some of these negative???
    dS0 = np.sum(IVIM_dS0(b, x[0], x_other[0], x[1], x_other[1]))
    dD = np.sum(-IVIM_dD(b, x[0], x_other[0], x[1], x_other[1]))
    return np.array([dS0, dD])


def IVIM_grad_lsqr_f_Dstar(x, b, s, x_other):
    df = IVIM_df(b, x_other[0], x[0], x_other[1], x[1])
    dDstar = IVIM_dDstar(b, x_other[0], x[0], x_other[1], x[1])
    return [df, dDstar]


def IVIM_grad_lsqr_f_Dstar_sumsq(x, b, s, x_other):
    # why are some of these negative???
    df = np.sum(-IVIM_df(b, x_other[0], x[0], x_other[1], x[1]))
    dDstar = np.sum(-IVIM_dDstar(b, x_other[0], x[0], x_other[1], x[1]))
    return np.array([df, dDstar])


#def IVIM_grad_lsqr_f_Dstar(x, b, s, x_other):
#    return IVIM_grad_wrap(np.array([x_other[0], x[0], x_other[1], x[1]]), b)[1:4:2]


def IVIM_fun_lsqr(x, b, s):
    return IVIM_fun_wrap(x, b) - s


def IVIM_fun_lsqr_sumsq(x, b, s):
    return np.sum(np.linalg.norm(IVIM_fun_lsqr(x, b, s)))
    # return np.sum(IVIM_fun_lsqr(x, b, s)**2.0)


def IVIM_fun_lsqr_S0_D(x, b, s, x_other):
    return IVIM_fun_wrap(np.array([x[0], x_other[0], x[1], x_other[1]]), b) - s


def IVIM_fun_lsqr_S0_D_sumsq(x, b, s, x_other):
    return np.sum(np.linalg.norm(IVIM_fun_lsqr_S0_D(x, b, s, x_other)))
    # return np.sum(IVIM_fun_lsqr_S0_D(x, b, s, x_other)**2.0)


def IVIM_fun_lsqr_f_Dstar(x, b, s, x_other):
    return IVIM_fun_wrap(np.array([x_other[0], x[0], x_other[1], x[1]]), b) - s


def IVIM_fun_lsqr_f_Dstar_sumsq(x, b, s, x_other):
    return np.sum(np.linalg.norm(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + 0.1*abs(x[0])+0.01*abs(x[1]-x_other[1])
    #return np.sum(np.linalg.norm(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + 0.1*np.sum(np.linalg.norm(np.array(x)-np.array([0,x_other[1]]))) #regularize, promote smoothness here
    # return np.sum(np.linalg.norm(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other))) + 0.1*np.linalg.norm(np.prod(np.array(x)-np.array([0,x_other[1]])))  # regularize on (f*(Dstar-D))^2
    # return np.sum(IVIM_fun_lsqr_f_Dstar(x, b, s, x_other)**2.0) + 0.01*np.sum(x**2.0) #regularize, promote smoothness here


def IVIM_fun(b, S0, f, D, Dstar):
    # this function evaluates the IVIM equation relative to S0, f, D, D* at the values b on the diffusion curve
    # should I be fitting S0????
    S = S0*(f*np.exp(-b*Dstar)+(1.0-f)*np.exp(-b*D))
    return S


def IVIM_grad(b, S0, f, D, Dstar):
    # this function evaluates the jacobian relative to S0, f, D, D* at the values b on the diffusion curve
    dS0 = IVIM_dS0(b, S0, f, D, Dstar)
    df = IVIM_df(b, S0, f, D, Dstar)
    dD = IVIM_dD(b, S0, f, D, Dstar)
    dDstar = IVIM_dDstar(b, S0, f, D, Dstar)
    return [dS0, df, dD, dDstar]


def IVIM_dS0(b, S0, f, D, Dstar):
    return f*np.exp(-b*Dstar)+(1.0-f)*np.exp(-b*D)


def IVIM_df(b, S0, f, D, Dstar):
    return S0*(np.exp(-b*Dstar)-np.exp(-b*D))  # +0.1*f/np.sqrt(Dstar**2.0+f**2.0)  # 2nd term 0.1*f... is regularization!


def IVIM_dD(b, S0, f, D, Dstar):
    return -S0*b*(1.0-f)*np.exp(-b*D)


def IVIM_dDstar(b, S0, f, D, Dstar):
    return -S0*b*f*np.exp(-b*Dstar)  # +0.1*Dstar/np.sqrt(Dstar**2.0+f**2.0)  # 2nd term 0.1*f... is regularization!


def IVIM_min_fun_S0_D(x, b, v, s, x_other):
    return IVIM_min_fun(np.array([x[0], x_other[0], x[1], x_other[1]]), b, v, s)


def IVIM_min_fun_f_Dstar(x, b, v, s, x_other):
    return IVIM_min_fun(np.array([x_other[0], x[0], x_other[1], x[1]]), b, v, s)


def IVIM_min_fun(x, b, v, s):
    # this function evaluates the fitness of an estimated fit x compared to actual values v evaluated at bvalues b with a sigma squared (noise level) of s
    # S0 = x[0]
    # f = x[1]
    # D = x[2]
    # Dstar = x[3]
    est = IVIM_fun(b, x[0], x[1], x[2], x[3])
    rpdf = ricepdf(v, est, s)
    # print(rpdf)
    rpdf[rpdf == 0] = 1e-100  # set all zero values to very very small values so they don't evaluate to -inf
    # if any((rpdf == 0)):  # finds if there are any zeros which we don't want to take the log of
    #    return np.finfo('d').max  # return a large number because we're minimizing
    # else:  # otherwise evaluate to True and run the computation
    return -np.sum(np.log(rpdf))  # sum of the log likelihood


def print_help():
    print('This program processes an IVIM scan and calculates the 4 parameters')
    print('These parameters are S0, f, Dstar, D')
    print('')
    print('Example:')
    print('%s -i <input file> -b <b-balue file> -f <output fit file name> ' %(str(sys.argv[0])))
    print('Input arguments:')
    print('-i <input file name of the 4D stacked image file>')
    print('-b <b-value file name of the csv file>')
    print('-s <standard deviation of the noise>')
    print('-m <mask image file name> (optional)')
    print('-l indicates the use of maximum likelihood fitting, by default it uses least squares (optional)')
    print('-1 indicates a one stage fitting (either -1 or -2 must be selected)')
    print('-2 indicates a two stage fitting (either -1 or -2 must be selected)')
    print('-e <minimization method, can be levenberg-marquardt or anything minimize accepts')
    print('-h prints this help dialog')
    print('Output arguments:')
    print('-f <output fit file name>')
    print('-n <output niterations file name> (optional)')
    print('-c <output success flag file name> (optional)')
    print('-u <output fitness file name> (optional)')
    print('-r <output residual file name> (optional)')
    print('-v <output curve file name> (optional)')

# use getopt to parse the input arguments
img_file_name = ''
bvalue_file_name = ''
noise_std = 0
mask_file_name = ''
output_fit_name = ''
output_niterations_name = ''
output_success_name = ''
output_fitness_name = ''
output_residual_name = ''
output_curve_name = ''
maximum_likelihood_fit = False
one_stage_fit = False
two_stage_fit = False
min_method = 'L-BFGS-B'
# use_jacobian = False
try:
    opts, args = go.getopt(sys.argv[1:], "hi:b:f:n:c:s:u:m:r:v:l12e:")
except go.GetoptError:
    print('Input option error')
    print_help()
    sys.exit(2)
for opt, arg in opts:
    print(opt, arg)
    if opt == '-h':
        print_help()
        sys.exit()
    elif opt == '-i':
        img_file_name = arg
    elif opt == '-b':
        bvalue_file_name = arg
    elif opt == '-s':
        noise_std = float(arg)
    elif opt == '-f':
        output_fit_name = arg
    elif opt == '-n':
        output_niterations_name = arg
    elif opt == '-c':
        output_success_name = arg
    elif opt == '-u':
        output_fitness_name = arg
    elif opt == '-m':
        mask_file_name = arg
    elif opt == '-r':
        output_residual_name = arg
    elif opt == '-v':
        output_curve_name = arg
    elif opt == '-l':
        maximum_likelihood_fit = True
    elif opt == '-1':
        one_stage_fit = True
    elif opt == '-2':
        two_stage_fit = True
    elif opt == '-e':
        min_method = arg
#    elif opt == '-j':
#        use_jacobian = True
    else:
        assert False, "Unhandled Option %s" % opt


if not img_file_name or not bvalue_file_name or not output_fit_name or noise_std == 0 or (not one_stage_fit and not two_stage_fit):  # we need at least 2 inputs and some reasonable outputs
    print('Missing one of the following:')
    print('Image file name: %s' % img_file_name)
    print('bvalue file name: %s' % bvalue_file_name)
    print('output fit file name: %s' % output_fit_name)
    print('noise std: %f' % noise_std)
    print('number of fit stages -1 or -2')
    print_help()
    sys.exit(2)


print('minimization method = %s' % min_method)
if maximum_likelihood_fit:
    print('Using maximum likelihood fitting (this is slow!)')
else:
    print('Using least squares fitting')

if one_stage_fit:
    print('Using one stage fitting')
if two_stage_fit:
    print('Using two stage fitting')

#if use_jacobian:
#    print('Using the jacobian ')


# load the 4-D stacked image
print('Reading the image')
hdr = nb.load(img_file_name)
shape = hdr.shape  # the image size
img = hdr.get_data().astype(np.dtype('d'))
shape3d = shape[0:-1]

# load the b-values
print('loading the b-values')
with open(bvalue_file_name, 'r') as bvalcsv:
    reader = csv.reader(bvalcsv)
    bvals_str = next(reader)[0].split(" ")
    bvals = np.array([float(i) for i in bvals_str])  # convert from strings to floats

if shape[3] != len(bvals):
    print("The 4th dimension of the image and the number of bvalues do not matach!")
    sys.exit(2)
else:
    print('we have %d b-values, so on y va!' % len(bvals))
    print(bvals)

# if we have a mask file input, then load the image
if mask_file_name:
    mask_hdr = nb.load(mask_file_name)
    mask_shape = mask_hdr.shape
    if shape3d != mask_shape:
        print('The mask and image are not the same shape')
        print(shape3d,mask_shape)
        sys.exit(2)
    mask = mask_hdr.get_data()
else:
    mask = np.ones(shape3d)



# set up some bounds and initial guesses
bnds = ((0.5, 1.5), (0, 1), (0, 0.5), (0, 0.5))  # (S0, f, D, D*)
bnds_S0_D = (bnds[0], bnds[2])
bnds_f_Dstar = (bnds[1], bnds[3])
# initial guesses are from the typical values in "Quantitative Measurement of Brin Perfusion with Intravoxel Incoherent Motion MR Imaging"
S0_guess = 1.0  # img.ravel().max()
f_guess = 6e-2  # 6% or 0.06
D_guess = 0.9e-3  # note that D<D* because the slope is lower in the diffusion section
Dstar_guess = 7.0e-3  # note that D*>D because the slope is higher in the perfusion section
# noise_std /= img.ravel().max()

bvals_le_200 = bvals[bvals<=200]
bvals_gt_200 = bvals[bvals>200]
b_le_200_cutoff = len(bvals_le_200)

# checking the gradient
print('checking the full gradient')
eps = np.sqrt(np.finfo('f').eps)
# fprime_approx = op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bvals[10])
# grdchk = op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bvals[10])
for bv in bvals:
    print(op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bv))
    print(IVIM_grad_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv))
    print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))

print('checking the partial gradient')
for bv in bvals:
    #print(op.approx_fprime([S0_guess, f_guess, D_guess, Dstar_guess], IVIM_fun_wrap, eps, bv))
    #print(IVIM_grad_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv))
    #print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))
    print('-')
    print(op.approx_fprime([S0_guess,D_guess],IVIM_fun_lsqr_S0_D_sumsq, eps, bv, IVIM_fun_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv), [f_guess, Dstar_guess]))
    print(IVIM_grad_lsqr_S0_D_sumsq([S0_guess, D_guess], bv, None, [f_guess, Dstar_guess]))
    print(op.approx_fprime([f_guess,Dstar_guess],IVIM_fun_lsqr_f_Dstar_sumsq, eps, bv, IVIM_fun_wrap([S0_guess, f_guess, D_guess, Dstar_guess], bv), [S0_guess, D_guess]))
    print(IVIM_grad_lsqr_f_Dstar_sumsq([f_guess, Dstar_guess], bv, None, [S0_guess, D_guess]))
    #print(op.check_grad(IVIM_fun_wrap, IVIM_grad_wrap, [S0_guess, f_guess, D_guess, Dstar_guess], bv))


print('Beginning processing')

# allocate before because I can't figure out how to allocate during
fit = np.zeros(list(shape3d)+[4, ])  # convert tuple to a list and append 4 to the end
nit = np.zeros(shape3d)
success = np.zeros(shape3d)
fun = np.zeros(shape3d)
residual = np.zeros(shape)
curve = np.zeros(shape)


# def mapprint(vec):
#     return vec
#
# def mapprintimg(pos,im):
#     return im[pos]
#
#
# print('parallel')
# pool=mp.Pool(8)
# test=pool.map(mapprint,shape3d,img)
# pool.close()
# print('end parallel')
#for item in np.ndindex(shape3d): #here's how to run a flat iterator!
#    print(item)
#    print(img[item])
#    print(img[item].shape)

#IVIM_Curve.IVIM_Curve.IVIM_fun_lsqr_S0_D_sumsq
#fit_S0_D = ic.IVIM_Curve(fun_in=ic.IVIM_fun_lsqr_S0_D_sumsq, method_in=min_method, bounds_in=bnds_S0_D)
#ic.IVIM_array_fit(fit_S0_D, [S0_guess, D_guess], img[:, :, :, b_le_200_cutoff:], mask, (bvals_gt_200, [0, 0]))

# just doing normal looping because nditer doesn't seem to really work when you have different sized arrays
for i in range(shape[0]):
    # print(i, shape[0], 100*i/shape[0])
    print('%d%%' % (100*i/shape[0]))
    for j in range(shape[1]):
        # print(i,j)
        for k in range(shape[2]):
            if (not any(img[i, j, k, :] == 0)) and (mask[i, j, k] > 0):  # if we have any zeros in the input then get rid of it
                # print(i,j,k)
                # S0_guess = img[i, j, k, :].max()
                im0 = img[i, j, k, 0]
                img[i, j, k, :] /= im0 #normalize to the first value
                if two_stage_fit:
                    if maximum_likelihood_fit or not min_method == 'levenberg-marquardt':
                        # first we fit S0 and D
                        if maximum_likelihood_fit:
                            res = op.minimize(IVIM_min_fun_S0_D, [S0_guess, D_guess], args=(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], noise_std, [0, 0]), bounds=bnds_S0_D, method=min_method)  # method='CG'
                        else:
                            res = op.minimize(IVIM_fun_lsqr_S0_D_sumsq, [S0_guess, D_guess], args=(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], [0, 0]), bounds=bnds_S0_D, method=min_method)  # , jac=IVIM_grad_lsqr_S0_D_sumsq)
                            # res = op.minimize(IVIM_fun_lsqr_S0_D_sumsq, [S0_guess, D_guess], args=(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], [0, 0]), bounds=bnds_S0_D, method=min_method, jac=IVIM_grad_lsqr_S0_D_sumsq)
                        fit[i, j, k, 0:3:2] = res.x  # vector of [S0, f, D, D*]
                        # then we fit f and D*
                        if maximum_likelihood_fit:
                            res = op.minimize(IVIM_min_fun_f_Dstar, [f_guess, Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff], noise_std, res.x), bounds=bnds_f_Dstar, method=min_method)
                        else:
                            res = op.minimize(IVIM_fun_lsqr_f_Dstar_sumsq, [f_guess, Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff], res.x), bounds=bnds_f_Dstar, method=min_method)  # , jac=IVIM_grad_lsqr_f_Dstar_sumsq)
                            # res = op.minimize(IVIM_fun_lsqr_f_Dstar_sumsq, [f_guess, Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff], res.x), bounds=bnds_f_Dstar, method=min_method, jac=IVIM_grad_lsqr_f_Dstar_sumsq)
                        fit[i, j, k, 1:4:2] = res.x  # vector of [S0, f, D, D*]
                        nit[i, j, k] = res.nit  # single value, number of iterations
                        success[i, j, k] = res.success  # single value: True, False. if it minimized well
                        fun[i, j, k] = res.fun  # single value, fitness function
                        residual[i, j, k, :] = img[i, j, k, :] - IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])
                        curve[i, j, k, :] = IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])*im0
                    else:
                        # x, cov_x = op.leastsq(IVIM_fun_lsqr_S0_D, [S0_guess, D_guess], args=(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], [0, 0]), full_output=False)
                        x, cov_x = op.leastsq(IVIM_fun_lsqr_S0_D, [S0_guess, D_guess], args=(bvals_gt_200, img[i, j, k, b_le_200_cutoff:], [0, 0]), Dfun=IVIM_grad_lsqr_S0_D, col_deriv=1, full_output=False)
                        fit[i, j, k, 0:3:2] = x
                        x, cov_x, infodict, mesg, ier = op.leastsq(IVIM_fun_lsqr_f_Dstar, [f_guess, Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff], x), Dfun=IVIM_grad_lsqr_f_Dstar, col_deriv=1, full_output=True)
                        # x, cov_x, infodict, mesg, ier = op.leastsq(IVIM_fun_lsqr_f_Dstar, [f_guess, Dstar_guess], args=(bvals_le_200, img[i, j, k, 0:b_le_200_cutoff], x), full_output=True)
                        fit[i, j, k, 1:4:2] = x  # vector of [S0, f, D, D*]
                        nit[i, j, k] = infodict['nfev']  # single value, number of iterations
                        success[i, j, k] = ier  # integer value which shows if the solution was found (1-4 are ok, otherwise not)
                        fun[i, j, k] = np.sum(np.linalg.norm(img[i, j, k, :] - IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])))  # single value, fitness function
                        residual[i, j, k, :] = img[i, j, k, :] - IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])
                        curve[i, j, k, :] = IVIM_fun(bvals, fit[i, j, k, 0], fit[i, j, k, 1], fit[i, j, k, 2], fit[i, j, k, 3])*im0
                if one_stage_fit:  # if it's a one stage fit
                    if maximum_likelihood_fit or not min_method == 'levenberg-marquardt':
                        if maximum_likelihood_fit:
                            res = op.minimize(IVIM_min_fun, [S0_guess, f_guess, D_guess, Dstar_guess], args=(bvals, img[i, j, k, :], noise_std), bounds=bnds, method=min_method)  # method='CG'
                        else:
                            res = op.minimize(IVIM_fun_lsqr_sumsq, [S0_guess, f_guess, D_guess, Dstar_guess], args=(bvals, img[i, j, k, :], noise_std), bounds=bnds, method=min_method, jac=IVIM_grad_lsqr_sumsq)
                        fit[i, j, k, :] = res.x  # vector of [S0, f, D, D*]
                        nit[i, j, k] = res.nit  # single value, number of iterations
                        success[i, j, k] = res.success  # single value: True, False. if it minimized well
                        fun[i, j, k] = res.fun  # single value, fitness function
                        residual[i, j, k, :] = img[i, j, k, :] - IVIM_fun(bvals, res.x[0], res.x[1], res.x[2], res.x[3])
                        curve[i, j, k, :] = IVIM_fun(bvals, res.x[0], res.x[1], res.x[2], res.x[3])*im0
                    else:
                        x, cov_x, infodict, mesg, ier = op.leastsq(IVIM_fun_lsqr, [S0_guess, f_guess, D_guess, Dstar_guess], args=(bvals, img[i, j, k, :]), Dfun=IVIM_grad_lsqr, col_deriv=1, full_output=True)
                        fit[i, j, k, :] = x  # vector of [S0, f, D, D*]
                        nit[i, j, k] = infodict['nfev']  # single value, number of iterations
                        success[i, j, k] = ier  # integer value which shows if the solution was found (1-4 are ok, otherwise not)
                        fun[i, j, k] = np.sum(np.linalg.norm(infodict['fvec'] - IVIM_fun(bvals, x[0], x[1], x[2], x[3])))  # single value, fitness function
                        residual[i, j, k, :] = img[i, j, k, :] - IVIM_fun(bvals, x[0], x[1], x[2], x[3])
                        curve[i, j, k, :] = IVIM_fun(bvals, x[0], x[1], x[2], x[3])*im0
                # print(res.nit)
                # print(img[i, j, k, :])

# output the images
# if output_fit_name:  # we always have this file because this check is done above
nb.save(nb.Nifti1Image(fit, hdr.get_affine()), output_fit_name)  # hdr.get_header()
if output_niterations_name:
    nb.save(nb.Nifti1Image(nit, hdr.get_affine()), output_niterations_name)  # hdr.get_header()
if output_success_name:
    nb.save(nb.Nifti1Image(success, hdr.get_affine()), output_success_name)  # hdr.get_header()
if output_fitness_name:
    nb.save(nb.Nifti1Image(fun, hdr.get_affine()), output_fitness_name)  # hdr.get_header()
if output_residual_name:
    nb.save(nb.Nifti1Image(residual, hdr.get_affine()), output_residual_name)
if output_curve_name:
    nb.save(nb.Nifti1Image(curve, hdr.get_affine()), output_curve_name)
