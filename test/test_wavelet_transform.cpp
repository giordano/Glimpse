/*
 * test_wavelet_transform.cpp
 *
 *  Created on: 27 June 2019
 *      Author: Timothy Spain
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "../src/wavelet_transform.h"

#define NPIX 1024
#define NSCALE 2

wavelet_transform* set_up() {

	int npix = NPIX;
	int nscale = NSCALE;
	// Generate a 2d wavelet transform
	wavelet_transform *wav = new wavelet_transform(npix, nscale, 1);

	return wav;
}

TEST_CASE( "Get nframes", "[wavelet_transform]" ) {
	wavelet_transform *wav = set_up();
	int frames_scale = 3; // wavelet_transform.cpp:47
	REQUIRE( wav->get_nframes() == NSCALE + frames_scale);

}

