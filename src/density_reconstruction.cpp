/*! Copyright CEA, 2016
 * author : Francois Lanusse < francois.lanusse@gmail.com >
 *
 * This software is a computer program whose purpose is to reconstruct mass maps
 * from weak gravitational lensing.
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */
#include <iostream>
#include <cmath>
#ifdef DEBUG_FITS
#include <sparse2d/IM_IO.h>
#endif

#include "density_reconstruction.h"

using namespace std;

density_reconstruction::density_reconstruction(boost::property_tree::ptree config, field *fi)
{
    f = fi;

    // Get options from the configuration file.
    nRecIter      = config.get<int>("parameters.niter", 500);
    nRecIterDebias= config.get<int>("parameters.niter_debias", 500);
    nscales       = config.get<int>("parameters.nscales", 4);
    lambda        = config.get<double>("parameters.lambda", 4.0);
    nrandom       = config.get<int>("parameters.nrandom", 1000.0);
    nreweights    = config.get<int>("parameters.nreweights", 5);
    positivity    = config.get<bool>("parameters.positivity", false);
    double bl_reg = config.get<double>("parameters.battle_lemarie_reg", 0.1);
    double ls_reg = config.get<double>("parameters.last_scale_reg", 0.1);


    cout << "Using following reconstruction parameters :" << endl;
    cout << "Number of scales         : " << nscales << endl;
    cout << "Regularisation parameter : " << lambda  << endl;

    // Get characteristics of the field
    npix = f->get_npix();
    nlp  = f->get_nlp();

    // Allocate wavelet transform
    wav = new wavelet_transform(npix, nscales, nlp);

    // Effective number of frames
    nframes = wav->get_nframes();

    ncoeff = npix * npix * nlp;
    nwavcoeff = npix * npix * nframes * nlp;

    // Allocating internal arrays
    delta       = fftwf_alloc_complex(ncoeff);
    delta_u     = fftwf_alloc_complex(ncoeff);
    delta_rec   = fftwf_alloc_complex(ncoeff);
    delta_old   = fftwf_alloc_complex(ncoeff);
    delta_grad  = fftwf_alloc_complex(ncoeff);
    delta_tmp   = fftwf_alloc_complex(ncoeff);
    delta_tmp_f = fftwf_alloc_complex(ncoeff);
    delta_trans = fftwf_alloc_complex(ncoeff);
    alpha       = (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_u     = (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_res   = (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_tmp   = (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_prox  = (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_prox_prev= (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_prox_old= (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_grad_old= (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_grad= (float *) malloc(sizeof(float) * nwavcoeff);
    alpha_rec   = (float *) malloc(sizeof(float) * nwavcoeff);
    thresholds  = (float *) malloc(sizeof(float) * nwavcoeff);
    weights     = (float *) malloc(sizeof(float) * nwavcoeff);
    support     = (float *) malloc(sizeof(float) * nwavcoeff);

    // Initialise internal arrays
    for (long ind = 0; ind < ncoeff; ind++) {
        delta[ind][0]     = 0;
        delta[ind][1]     = 0;
        delta_u[ind][0]   = 0;
        delta_u[ind][1]   = 0;
        delta_old[ind][0] = 0;
        delta_old[ind][1] = 0;
        delta_rec[ind][0] = 0;
        delta_rec[ind][1] = 0;
        delta_grad[ind][0] = 0;
        delta_grad[ind][1] = 0;
        delta_tmp[ind][0] = 0;
        delta_tmp[ind][1] = 0;
        delta_tmp_f[ind][0] = 0;
        delta_tmp_f[ind][1] = 0;
    }

    for (long ind = 0; ind < nwavcoeff; ind++) {
        alpha[ind]     = 0;
        alpha_u[ind]   = 0;
        alpha_res[ind] = 0;
        alpha_rec[ind] = 0;
        alpha_prox[ind] = 0;
        alpha_prox_prev[ind] = 0;
        alpha_prox_old[ind] = 0;
        alpha_grad[ind] = 0;
        alpha_grad_old[ind] = 0;
        alpha_tmp[ind] = 0;
        thresholds[ind] = 0;
        weights[ind]   = 1;
        support[ind]   = 1;
    }

    // Initialise the proximal operator
    #ifdef CUDA_ACC
    prox = new spg(npix, nlp, nframes, f->get_preconditioning_matrix(), thresholds);
    #endif

    // Normalization factor for the fft
    fftFactor     = 1.0 / (((double)npix) * npix);

    int dimensions[2] = { npix, npix };
    int rank = 2;


#ifdef CUDA_ACC
    // Retrieve which GPUs to use
    getDeviceCount(&nGPU);
    getGPUs(whichGPUs);

    cufftResult ret = cufftCreate(&fft_plan);

    if (nGPU > 1) {
        ret = cufftXtSetGPUs(fft_plan , nGPU, whichGPUs);
        if (ret != 0) std::cout << "set gpus" << ret << std::endl;

        ret = cufftMakePlanMany(fft_plan, rank, dimensions,
                                NULL, 1, npix * npix, NULL, 1, npix * npix,
                                CUFFT_C2C, nlp, worksize);
        if (ret != 0) std::cout << "Make plan " << ret << std::endl;

        ret = cufftXtMalloc(fft_plan, &d_frameXt, CUFFT_XT_FORMAT_INPLACE);
        if (ret != 0) std::cout << "malloc " << ret << std::endl;
    } else {
        ret = cufftMakePlanMany(fft_plan, rank, dimensions,
                                NULL, 1, npix * npix, NULL, 1, npix * npix,
                                CUFFT_C2C, nlp, worksize);
        if (ret != 0) std::cout << "make plan " << ret << std::endl;

        cudaMalloc(&d_frame, sizeof(cufftComplex)*nlp * npix * npix);

    }
#else
    fft_frame = fftwf_alloc_complex(npix * npix * nlp);
    plan_forward = fftwf_plan_many_dft(rank, dimensions, nlp ,
                                      fft_frame, NULL, 1, npix*npix,
                                      fft_frame, NULL, 1, npix*npix,
                                      FFTW_FORWARD, FFTW_MEASURE);
    plan_backward = fftwf_plan_many_dft(rank, dimensions, nlp,
                                       fft_frame, NULL, 1, npix*npix,
                                       fft_frame, NULL, 1, npix*npix,
                                       FFTW_BACKWARD, FFTW_MEASURE);
#endif


    // Initialize the threshold levels, with lower thresholds on larger scales
    sigma_thr = (double *) malloc(sizeof(double) * nframes);
    for (int i = 0; i < nscales - 1; i++) {
        sigma_thr[i] = lambda * sqrt(2 * log(npix / pow(2.0, i) * npix / pow(2.0, i))) / sqrt(2 * log(npix * npix));
    }

    // Special regularisation for the smooth approximation
    sigma_thr[nscales - 1] = ls_reg;

    // Additional regularisation for additional BL frames
    for (int i = nscales; i < nframes; i++) {
        sigma_thr[i] = bl_reg;
    }

    mu1 = 1.0;
    mu2 = f->get_spectral_norm(200, 1e-5);
    tau = 0.2 / mu2;
    sig = 0.45*(1.0 / tau - mu2/2.0);
    std::cout << "Tau " << tau << " Sigma " << sig << std::endl;
}

density_reconstruction::~density_reconstruction()
{
    fftwf_free(delta);
    fftwf_free(delta_u);
    fftwf_free(delta_rec);
    fftwf_free(delta_old);
    fftwf_free(delta_grad);
    fftwf_free(delta_tmp);
    fftwf_free(delta_tmp_f);
    fftwf_free(delta_trans);
    free(sigma_thr);
    free(alpha);
    free(alpha_u);
    free(alpha_res);
    free(alpha_tmp);
    free(alpha_prox);
    free(alpha_prox_prev);
    free(alpha_prox_old);
    free(alpha_grad);
    free(alpha_grad_old);
    free(thresholds);
    free(weights);

#ifdef CUDA_ACC
    if (nGPU > 1) {
        cufftXtFree(d_frameXt);
    } else {
        cudaFree(d_frame);
    }
    cufftDestroy(fft_plan);

    delete prox;
#else
    fftwf_destroy_plan(plan_backward);
    fftwf_destroy_plan(plan_forward);
    fftwf_free(fft_frame);
#endif

}


void density_reconstruction::direct_fourier_transform(fftwf_complex *input, fftwf_complex *output)
{

#ifdef CUDA_ACC
    if (nGPU > 1) {
        cufftXtMemcpy(fft_plan, d_frameXt, input, CUFFT_COPY_HOST_TO_DEVICE);
        cufftXtExecDescriptorC2C(fft_plan, d_frameXt, d_frameXt, CUFFT_FORWARD);
        cufftXtMemcpy(fft_plan, output, d_frameXt, CUFFT_COPY_DEVICE_TO_HOST);
    } else {
        cudaMemcpy(d_frame, input, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyHostToDevice);
        cufftExecC2C(fft_plan, d_frame, d_frame, CUFFT_FORWARD);
        cudaMemcpy(output, d_frame, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyDeviceToHost);
    }
#else

    #pragma omp parallel for
    for (long ind = 0; ind < ncoeff; ind++) {
       fft_frame[ind][0] = input[ind][0]; fft_frame[ind][1] = input[ind][1];
    }
    fftwf_execute(plan_forward);
    #pragma omp parallel for
    for (long ind = 0; ind < ncoeff; ind++) {
        output[ind][0] = fft_frame[ind][0]; output[ind][1] = fft_frame[ind][1];
    }
#endif
}


void density_reconstruction::inverse_fourier_transform(fftwf_complex *input, fftwf_complex *output)
{
#ifdef CUDA_ACC
    if (nGPU > 1) {
        cufftXtMemcpy(fft_plan, d_frameXt, input, CUFFT_COPY_HOST_TO_DEVICE);
        cufftXtExecDescriptorC2C(fft_plan, d_frameXt, d_frameXt, CUFFT_INVERSE);
        cufftXtMemcpy(fft_plan, output, d_frameXt, CUFFT_COPY_DEVICE_TO_HOST);
    } else {
        cudaMemcpy(d_frame, input, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyHostToDevice);
        cufftExecC2C(fft_plan, d_frame, d_frame, CUFFT_INVERSE);
        cudaMemcpy(output, d_frame, sizeof(cufftComplex)* npix * npix * nlp, cudaMemcpyDeviceToHost);
    }
#else

    #pragma omp parallel for
    for (long ind = 0; ind < ncoeff; ind++) {
        fft_frame[ind][0] = input[ind][0]; fft_frame[ind][1] = input[ind][1];
    }
    fftwf_execute(plan_backward);
    #pragma omp parallel for
    for (long ind = 0; ind < ncoeff; ind++) {
        output[ind][0] = fft_frame[ind][0]; output[ind][1] = fft_frame[ind][1];
    }
#endif
}


void density_reconstruction::run_main_iteration(long int niter, bool debias)
{

#ifdef DEBUG_FITS
    dblarray rec_delta(f->get_npix(), f->get_npix(), f->get_nlp());
    char name[256];
#endif

    double old_tk = 1.0;
    double tk;

    for (long iter = 0; iter < niter; iter++) {
        std::cout << "Iteration : " << iter << std::endl;

        // Compute gradient
        std::memcpy(delta_grad, delta, sizeof(fftwf_complex) * ncoeff);
        f->gradient(delta_grad);

        // Compute adjoint of the wavelet transform
        wav->trans_adjoint(alpha_u, delta_u);

        // Updating delta by a gradient step
            #pragma omp parallel for
            for (long ind = 0; ind < ncoeff; ind++) {
                delta_tmp[ind][0] = delta[ind][0] + tau * (delta_grad[ind][0] - delta_u[ind][0]);
                delta_tmp[ind][1] = delta[ind][1] + tau * (delta_grad[ind][1] - delta_u[ind][1]);
            }

        // Apply positivity and/or E mode constraint
        inverse_fourier_transform(delta_tmp, delta_tmp);
        if(positivity){
             #pragma omp parallel for
            for (long ind = 0; ind < ncoeff; ind++) {
                alpha_tmp[ind] = delta_tmp[ind][0] * fftFactor;
            }

#ifdef CUDA_ACC
            prox->prox_pos(alpha_tmp, 10000, iter == niter/2);
#else

#endif
            #pragma omp parallel for
            for (long ind = 0; ind < ncoeff; ind++) {
                delta_tmp[ind][0] = delta_tmp[ind][0] * fftFactor - alpha_tmp[ind];
                delta_tmp[ind][1] = 0;
            }
        }else{
            #pragma omp parallel for
            for (long ind = 0; ind < ncoeff; ind++) {
                delta_tmp[ind][0] *= fftFactor;
                delta_tmp[ind][1] = 0;
            }
        }
        direct_fourier_transform(delta_tmp, delta_tmp);
        ///////////////////////////////////////////////////////


        #pragma omp parallel for
        for (long ind = 0; ind < ncoeff; ind++) {
            delta_tmp_f[ind][0] = 2 * delta_tmp[ind][0] - delta[ind][0];
            delta_tmp_f[ind][1] = 2 * delta_tmp[ind][1] - delta[ind][1];
        }

        wav->transform(delta_tmp_f, alpha_tmp);

        #pragma omp parallel for
        for (long ind = 0; ind < nwavcoeff; ind++) {
          alpha_tmp[ind] = alpha_u[ind] + sig * alpha_tmp[ind];
        }

#ifdef CUDA_ACC
        prox->prox_l1(alpha_tmp,1000, iter == niter/2);
#else
        // TODO: Implement CPU prox operator
#endif

        // Fista update of the iterates
        tk = 0.5 * (1.0 + sqrt(1.0 + 4.0 * old_tk * old_tk));
        // Updating delta
        #pragma omp parallel for
        for (long ind = 0; ind < ncoeff; ind++) {
            delta[ind][0] = delta_tmp[ind][0] + (old_tk - 1.) / tk * (delta_tmp[ind][0] - delta[ind][0]);
            delta[ind][1] = delta_tmp[ind][1] + (old_tk - 1.) / tk * (delta_tmp[ind][1] - delta[ind][1]);
        }
        // Updating alpha
        #pragma omp parallel for
        for (long ind = 0; ind < nwavcoeff; ind++) {
            alpha_u[ind] = alpha_tmp[ind] + (old_tk - 1.) / tk * (alpha_tmp[ind] - alpha_u[ind]);
        }

        old_tk = tk;
#ifdef DEBUG_FITS
        get_density_map(rec_delta.buffer());
        if(iter % 10 == 0){
            sprintf(name, "delta_%03d.fits", iter);
        }else{
            sprintf(name, "delta_tmp.fits", iter);
        }
        fits_write_dblarr(name, rec_delta);
#endif
    }
}

void density_reconstruction::analysis_prox(fftwf_complex *delta_in)
{
    int niter =500;
    double tau_prox = 0.001;
    double old_tk = 1.0;
    double tk = 1.0;
    float opt0=1.0;
    float opt=0;

    // Corrects for the preconditioning matrix
    const double *P = f->get_preconditioning_matrix();

    for (int iter = 0; iter < niter; iter ++) {

        // Compute gradient
        wav->trans_adjoint(alpha_prox, delta_tmp_f);

        for (int z = 0; z < nlp; z++) {
            for (long ind = 0; ind < npix*npix ; ind++) {
                delta_trans[z * npix * npix + ind][0] = 0;
                delta_trans[z * npix * npix + ind][1] = 0;

                for (int z2 = 0; z2 < nlp; z2++) {
                    delta_trans[z * npix * npix + ind][0] += P[z * nlp + z2] * delta_tmp_f[z2 * npix * npix + ind][0];
                    delta_trans[z * npix * npix + ind][1] += P[z * nlp + z2] * delta_tmp_f[z2 * npix * npix + ind][1];
                }
            }
        }

        #pragma omp parallel for
        for (long ind = 0; ind < ncoeff; ind++) {
            delta_trans[ind][0] = delta_in[ind][0] / tau - delta_trans[ind][0];
            delta_trans[ind][1] = delta_in[ind][1] / tau - delta_trans[ind][1];
        }

        for (int z = 0; z < nlp; z++) {
            for (long ind = 0; ind < npix*npix ; ind++) {
                delta_tmp_f[z * npix * npix + ind][0] = 0;
                delta_tmp_f[z * npix * npix + ind][1] = 0;

                for (int z2 = 0; z2 < nlp; z2++) {
                    delta_tmp_f[z * npix * npix + ind][0] += P[z * nlp + z2] * delta_trans[z2 * npix * npix + ind][0];
                    delta_tmp_f[z * npix * npix + ind][1] += P[z * nlp + z2] * delta_trans[z2 * npix * npix + ind][1];
                }
            }
        }


        wav->transform(delta_tmp_f, alpha_grad);

        // Compute BB update
        double ss=0;
        double sy=0;
        double yy=0;
        double s=0;
        double y=0;
        for (long ind = 0; ind < nwavcoeff; ind++) {
            s = (alpha_prox[ind] - alpha_prox_old[ind]);
            alpha_prox_old[ind] = alpha_prox[ind];
            y = (alpha_grad[ind] - alpha_grad_old[ind]);
            alpha_grad_old[ind] = alpha_grad[ind];
            ss += s * s;
            yy += y * y;
            sy += s * y;
        }

        if(iter > 0){
            if(iter % 2)
                tau_prox = -sy/yy;
            else
                tau_prox = -ss/sy;
        }
        opt = 0;
        for (long ind = 0; ind < nwavcoeff; ind++) {

            double dum = alpha_prox[ind];
            double w = weights[ind];
            double g = alpha_grad[ind];
            double o;
            // Compute optimality check
            if( dum >=  w*(1.0 - 1e-5)){
                o = fabsf(fmaxf(0.0f, -g)) / fmaxf(w, 1e-5f);
            }else  if( dum <= -w*(1.0 - 1e-5)) {
                o = fabsf(fmaxf(0.0f, g)) / fmaxf(w, 1e-5f);
            }else{
                o = fabsf(g) / fmaxf(w, 1e-5f);
            }
            opt = o > opt ? o : opt;
            // Gradient descent
            dum = alpha_prox[ind] + tau_prox * alpha_grad[ind];

            // Projection
            double val = dum - copysign(max(fabs(dum) - weights[ind], 0.0), dum);
            alpha_prox[ind] = val;
        }

        opt0 = opt < opt0 ? opt : opt0;
        if (iter  % 100 == 0) std::cout << iter << " opt : " << opt << " ; tau : " << tau_prox <<std::endl;

        if((opt < old_opt) || (opt < 5e-3) || ((iter >= 0.5*niter) && (opt == opt0) )){
            std::cout << iter << " opt : " << opt << " ; tau : " << tau_prox <<std::endl;
            break;
        }

    }

    old_opt = opt0;

    // Return prox
    wav->trans_adjoint(alpha_prox, delta_tmp_f);

    for (int z = 0; z < nlp; z++) {
        for (long ind = 0; ind < npix*npix ; ind++) {
            delta_trans[z * npix * npix + ind][0] = 0;
            delta_trans[z * npix * npix + ind][1] = 0;

            for (int z2 = 0; z2 < nlp; z2++) {
                delta_trans[z * npix * npix + ind][0] += P[z * nlp + z2] * delta_tmp_f[z2 * npix * npix + ind][0];
                delta_trans[z * npix * npix + ind][1] += P[z * nlp + z2] * delta_tmp_f[z2 * npix * npix + ind][1];
            }
        }
    }

    for (long ind = 0; ind < ncoeff; ind++) {
        delta_in[ind][0] = delta_in[ind][0] - delta_trans[ind][0] * tau;
        delta_in[ind][1] = delta_in[ind][1] - delta_trans[ind][1] * tau;
    }
}


void density_reconstruction::reconstruct()
{
    std::cout << "Computing thresholds" << std::endl;

    compute_thresholds(nrandom);

    for (long z = 0 ; z < nlp ; z++) {
        long offset = z * nframes * npix * npix;
        for (long n = 0; n < nframes; n++) {
            for (long ind = 0; ind < npix * npix; ind++) {
                weights[offset + n * npix * npix + ind] = sigma_thr[n] * thresholds[offset + n * npix * npix + ind];
            }
        }
    }

#ifdef DEBUG_FITS
    // Saves the thresholds
    fltarray thr;
    thr.alloc(weights, npix,npix,nframes*nlp);
    fits_write_fltarr("thresholds.fits", thr);
#endif

#ifdef CUDA_ACC
    prox->update_weights(weights);
#endif

    std::cout << "Running main iteration" << std::endl;
    run_main_iteration(nRecIter);

    // Reweighted l1 loop
    for (int i = 0; i < nreweights ; i++) {
            f->update_covariance(delta);
            compute_thresholds(nrandom / 2);
            compute_weights();
            run_main_iteration(nRecIter / 2);
        }

    std::cout  << "Starting debiasing " << std::endl;
    // Final debiasing step
    f->update_covariance(delta);
    run_main_iteration(nRecIterDebias, true);
}

void density_reconstruction::compute_thresholds(int niter)
{

    for (long ind = 0; ind < nwavcoeff; ind++) {
        thresholds[ind] = 0;
    }

    for (int i = 0; i < niter ; i++) {
        f->gradient_noise(delta_rec);
        f->combine_components(delta_rec, delta_trans);

        wav->transform(delta_trans, alpha_tmp);

        // Compute gradient step
        for (long ind = 0; ind < nwavcoeff; ind++) {

            thresholds[ind] += pow(alpha_tmp[ind], 2.0);
        }
    }
    for (long z = 0 ; z < nlp ; z++) {
        long offset = z * nframes * npix * npix;
        for (long n = 0; n < nframes; n++) {
            double maxThr = 0;
            for (long ind = 0; ind < npix * npix; ind++) {
                thresholds[offset + n * npix * npix + ind] = sqrt(1.0 / ((double) niter) * thresholds[offset + n * npix * npix + ind]);
                maxThr = thresholds[offset + n * npix * npix + ind] > maxThr ? thresholds[offset + n * npix * npix + ind] : maxThr;
            }
            for (long ind = 0; ind < npix * npix; ind++) {
                thresholds[offset + n * npix * npix + ind] = max(thresholds[offset + n * npix * npix + ind], (float) maxThr * 0.1f);
            }
        }
    }
}

double density_reconstruction::get_spectral_norm_prox(int niter, double tol)
{

    double norm = 0;
    double norm_old = 0;

    // Initialise array with random numbers
    norm = 0;
    f->gradient_noise(delta_tmp);
    for (long ind = 0; ind < ncoeff; ind++) {
        norm += delta_tmp[ind][0] * delta_tmp[ind][0] + delta_tmp[ind][1] * delta_tmp[ind][1];
    }
    norm = sqrt(norm);

    // Normalise the input
    for (long ind = 0; ind < ncoeff; ind++) {
        delta_tmp[ind][0] /= norm;
        delta_tmp[ind][1] /= norm;
    }
    for (int k = 0; k < niter; k++) {

        wav->transform(delta_rec, alpha_u);
        wav->trans_adjoint(alpha_u, delta_rec);

        // Compute norm
        for (long ind = 0; ind < ncoeff; ind++) {
            norm += delta_tmp[ind][0] * delta_tmp[ind][0] + delta_tmp[ind][1] * delta_tmp[ind][1];
        }
        norm = sqrt(norm);


        if (fabs(norm - norm_old) / norm <= tol) {
            break;
        }

        for (long ind = 0; ind < ncoeff; ind++) {
            delta_tmp[ind][0] /= norm;
            delta_tmp[ind][1] /= norm;
        }

        norm_old = norm;
    }

    return norm * (1.0 + tol);
}


void density_reconstruction::compute_weights()
{
    for (long ind = 0; ind < ncoeff; ind++) {
        delta_tmp[ind][0] = delta[ind][0] * fftFactor;
        delta_tmp[ind][1] = delta[ind][1] * fftFactor;
    }
    inverse_fourier_transform(delta_tmp, delta_tmp);
    for (long ind = 0; ind < ncoeff; ind++) {
        delta_tmp[ind][0] = max(delta_tmp[ind][0], 0.0f);
        delta_tmp[ind][1] = 0;
    }
    direct_fourier_transform(delta_tmp, delta_tmp);

    wav->transform(delta_tmp, alpha_res);

    for (long ind = 0; ind < nwavcoeff; ind++) {
        if (fabs(alpha_res[ind]) < lambda * thresholds[ind]) {
            weights[ind] = 1.0;
        } else {
            weights[ind] = lambda * thresholds[ind] / fabs(alpha_res[ind]);
        }
    }
}

void density_reconstruction::get_density_map(double *d)
{

    for (long ind = 0; ind < ncoeff; ind++) {
        delta_tmp[ind][0] = delta[ind][0] * fftFactor;
        delta_tmp[ind][1] = delta[ind][1] * fftFactor;
    }
    inverse_fourier_transform(delta_tmp, delta_tmp);

    // Corrects for the preconditioning matrix
    const double *P = f->get_preconditioning_matrix();

    for (int z = 0; z < nlp; z++) {
        for (int y = 0; y < npix ; y++) {
            for (int x = 0; x < npix ; x++) {
                long pos = (npix - y - 1) * npix + (npix - x - 1);
                d[z * npix * npix + x * npix + y] = 0;
                for (int z2 = 0; z2 < nlp; z2++) {
                    d[z * npix * npix + x * npix + y] += P[z * nlp + z2] * delta_tmp[z2 * npix * npix + pos][0];
                }
            }

        }
    }
}
