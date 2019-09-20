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

#ifndef DENSITY_RECONSTRUCTION_H
#define DENSITY_RECONSTRUCTION_H

#include <boost/property_tree/ptree.hpp>

#include "field.h"
#ifdef pi
#undef pi
#endif


#ifdef CUDA_ACC
#include <cufftXt.h>
#include "spg.h"
#include "gpu_utils.h"
#endif

#ifndef CAPTURE_OUTPUT
#define CAPTURE_OUTPUT true
#endif

#include "wavelet_transform.h"


class density_reconstruction
{
    // Reconstruction parameters
    int    nscales;                     /*!< Number of dyadic wavelet scales. */
    int    nRecIter;                    /*!< Number of main iterations. */
    int    nRecIterDebias;              /*!< Number of iterations for the debiasing. */
    double lambda;                      /*!< Regularisation parameter. */
    int    nreweights;                  /*!< Number of reweighted l1 iterations.*/
    bool   positivity;                  /*!< Apply positivity constraint on the reconstruction. */
    
    // Internal parameters
    int npix;                           /*!< Number of pixels. */
    int nlp;                            /*!< Number of lens planes. */
    int ncoeff;                         /*!< Number of reconstruction coefficients. */
    int nframes;                        /*!< Total number of wavelet frames. */
    int nwavcoeff;                      /*!< Number of wavelet coefficients. */
    int nrandom;                        /*!< Number of noise randomisations for building thresholds. */
    double * sigma_thr;                 /*!< Array storing the regularisation parameter for each wavelet frame. */
    double mu1, mu2, sig, tau;          /*!< Hyper-parameters for the algorithm. */
    float old_opt;

    wavelet_transform *wav;             /*!< Wavelet transform operator. */
    field *f;                           /*!< Lensing field to use for the reconstruction.*/
#ifdef CUDA_ACC
    spg   *prox;                        /*!< SPG algorithm for fast evaluation of simple proximal operators */
#endif
    
    // Internal reconstruction arrays
    fftwf_complex * delta;
    fftwf_complex * delta_rec;
    fftwf_complex * delta_u;
    fftwf_complex * delta_old;
    fftwf_complex * delta_grad;
    fftwf_complex * delta_tmp;
    fftwf_complex * delta_tmp_f;
    fftwf_complex * delta_trans;
    float * alpha;                     
    float * alpha_u;
    float * alpha_res;
    float * alpha_rec;
    float * alpha_tmp;
    float * alpha_grad_old;
    float * alpha_grad;
    float * alpha_prox;
    float * alpha_prox_prev;
    float * alpha_prox_old;
    float * thresholds;
    float * support;
    float * weights;
    
    double fftFactor;
    
#ifdef CUDA_ACC
    int nGPU;
    int whichGPUs[MAX_GPUS];
    size_t worksize[MAX_GPUS];
    cudaLibXtDesc *d_frameXt;
    cufftHandle  fft_plan;
    cufftComplex  *d_frame;
#else
    fftwf_complex *fft_frame;
    fftwf_plan plan_forward, plan_backward;
#endif

    double get_spectral_norm_prox(int niter, double tol);
    
    void direct_fourier_transform(fftwf_complex *in, fftwf_complex *out);
    void inverse_fourier_transform(fftwf_complex *in, fftwf_complex *out);
    
    void analysis_prox(fftwf_complex* delta_in);

public:
    
    /*! Initialise 3D density reconstruction algorithm.
     * 
     */
    density_reconstruction(boost::property_tree::ptree config, field* f);

    /*! Destructor.
     * 
     */
    ~density_reconstruction();
    
     /*! Run the main iteration of the reconstruction algorithm for a number of iterations.
     * 
     */
    void run_main_iteration(long niter, bool debias=false);
    
    /*! Performs the reconstruction
     * 
     */
    void reconstruct();
    
    /*! Compute the noise thresholds.
     * 
     */
    void compute_thresholds(int niter);
    
    /*! Update weights for reweighted-l1 based on current solution.
     * 
     */
    void compute_weights();
    
    /*! Get current reconstructed density map
     * 
     */
    void get_density_map(double *density);

private:
    void write_l1_weights(char* suffix);
    
};

#endif // DENSITY_RECONSTRUCTION_H
