/*
 * test_field.cpp
 *
 *  Created on: 27 June 2019
 *      Author: Timothy Spain, t.spain@ucl.ac.uk
 */

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <string>
#include <sstream>

#include <boost/property_tree/ini_parser.hpp>

#include "../src/field.h"
#include "../src/survey.h"

void coord_test(int npix, int i, int j, double ra_array[], double de_array[], double ra_target, double de_target, double acc);

class mock_survey : public survey {

	public :
//	mock_survey() : survey(new boost::property_tree::ptree()) {
//
//	}
	void load(std::string fileName) {
		// does nothing
	}
	long get_ngal() {
		return 1;
	}
	bool get_flexion_availability() {
		return false;
	}
	// A 0.1 radian survey around Sgr A*
	double get_size() {
		return 0.1;
	}
	double get_center_dec() {
		return -0.5;
	}
	double get_center_ra() {
		return 4.6;
	}
	double get_gamma1(long gal_index) {
		return 0.0;
	}
	double get_gamma2(long gal_index) {
		return 0.0;
	}
	double get_F1(long gal_index) {
		return 0.;
	}
	double get_F2(long gal_index) {
		return 0.;
	}
	double get_ra(long gal_index) {
		return 4.649850989853492;
	}
	double get_dec(long gal_index) {
		return 0.50628180298921;
	}
	double get_shear_weight(long gal_index) {
		return 0.;
	}
	double get_flexion_weight(long gal_index) {
		return 0.;
	}
	redshift_distribution *get_redshift(long gal_index) {
		// watch out for code using malloc/free rather than new/delete
		return new photometric_redshift(0., 0., 0.); // The Milky Way is definitely at zero red shift
	}
};



std::string ini = "[survey]\n"
			"center_ra=4.6\n"
			"center_dec=-0.5\n"
			"size=0.1\n"
			"units=radian\n"
			"\n"
			"[cosmology]\n"
			"Omega_m=0.25\n"
			"h=0.70\n"
			"\n"
			"[field]\n"
			"units=radian\n"
			"pixel_size=0.001\n"
			"padding=28\n"
			"include_flexion=false\n"
			"zlens=0.3\n";

int npix = (int) 0.001/0.1 + 0.5;

field *set_up( ) {
	auto pt = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini(ini, pt);
	field *ff = new field(*pt, (survey*) new mock_survey());

	return ff;
}

TEST_CASE( "Test pixel coordinates for the mock field", "[field]") {
	field *ff = set_up();

	double *ra = new double[npix*npix];
	double *de = new double[npix*npix];

	ff->get_pixel_coordinates(ra, de);

	// Externally calculated using python
	double ra00 = 260.2427115353536;
	double de00 = -31.438926679481305;
	double ra099 = 260.41707040776555;
	double de099 = -25.78025687329525;
	double ra990 = 266.87845998500376;
	double de990 = -31.438926679481305;
	double ra9999 = 266.7041011125918;
	double de9999 = -25.78025687329525;

	double acc = 1e-8;

	coord_test(npix, 0, 0, ra, de, ra00, de00, acc);
	coord_test(npix, 0, 99, ra, de, ra099, de099, acc);
	coord_test(npix, 99, 0, ra, de, ra990, de990, acc);
	coord_test(npix, 99, 99, ra, de, ra9999, de9999, acc);

}

void coord_test(int npix, int i, int j, double ra_array[], double de_array[], double ra_target, double de_target, double acc) {
	int idx = i*npix + j;
	double delta_ra = std::abs(ra_array[idx] - ra_target);
	double delta_de = std::abs(de_array[idx] - de_target);

	REQUIRE( ((delta_ra < acc) && (delta_de < acc)) );

}
