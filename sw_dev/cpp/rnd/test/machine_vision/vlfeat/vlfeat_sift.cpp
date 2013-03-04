#define VL_SIFT_DRIVER_VERSION 0.1

#include "generic-driver.h"

#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/sift.h>
#include <vl/getopt_long.h>

#include <iostream>
#include <string>


namespace {
namespace local {

bool read_pgm(const std::string &name, vl_sift_pix *& fdata, VlPgmImage &pim, const bool verbose)
{
	// open input file
	FILE *in = fopen(name.c_str(), "rb");
	if (!in)
	{
		std::cerr << "file not found" << std::endl;
		return false;
	}

	// read data

	// read PGM header
	vl_bool err = vl_pgm_extract_head(in, &pim);
	if (err)
	{
		switch (vl_get_last_error())
		{
		case  VL_ERR_PGM_IO:
			std::cerr << "cannot read from " << name << std::endl;
			return false;
		case VL_ERR_PGM_INV_HEAD:
			std::cerr << name << " contains a malformed PGM header." << std::endl;
			return false;
		}
	}

	if (verbose)
		std::cout << "sift: image is " << pim.width << " by " << pim.height << " pixels" << std::endl;

	// allocate buffer
	vl_uint8 *data  = new vl_uint8 [vl_pgm_get_npixels(&pim) * vl_pgm_get_bpp(&pim)];
	fdata = new vl_sift_pix [vl_pgm_get_npixels(&pim) * vl_pgm_get_bpp(&pim)];
	if (!data || !fdata)
	{
		std::cerr << "could not allocate enough memory." << std::endl;
		return false;
	}

	// read PGM body
	err = vl_pgm_extract_data(in, &pim, data) ;
	if (err)
	{
		std::cerr << "PGM body malformed." << std::endl;
		return false;
	}

	// convert data type
	for (unsigned q = 0 ; q < unsigned(pim.width * pim.height) ; ++q)
	{
		fdata[q] = data[q];
	}

	fclose(in);

	delete [] data;
	data = NULL;

	return true;
}

int save_gss(VlSiftFilt *filt, VlFileMeta *fm, const char *basename, bool verbose)
{
	if (!fm -> active)
		return VL_ERR_OK;

	const int w = vl_sift_get_octave_width(filt);
	const int h = vl_sift_get_octave_height(filt);

	VlPgmImage pim;
	pim.width = w;
	pim.height = h;
	pim.max_value = 255;
	pim.is_raw = 1;

	vl_uint8 *buffer = new vl_uint8 [w * h];
	if (!buffer)
		return VL_ERR_ALLOC;

	char tmp[1024];
	const vl_size q = vl_string_copy(tmp, sizeof(tmp), basename) ;
	if (q >= sizeof(tmp))
	{
		delete [] buffer;
		return VL_ERR_OVERFLOW;
	}

	int S = filt->S;
	int o = filt->o_cur;
	int err;
	for (int s = 0 ; s < S ; ++s)
	{
		vl_sift_pix *pt = vl_sift_get_octave(filt, s);

		// conversion
		for (int i = 0 ; i < w * h ; ++i)
			buffer[i] = (vl_uint8)pt[i];

		// save
		snprintf(tmp + q, sizeof(tmp) - q, "_%02d_%03d", o, s);

		err = vl_file_meta_open(fm, tmp, "wb");
		if (err)
		{
			delete [] buffer;
			vl_file_meta_close(fm);
			return err;
		}

		err = vl_pgm_insert(fm->file, &pim, buffer);
		if (err)
		{
			delete [] buffer;
			vl_file_meta_close(fm);
			return err;
		}

		if (verbose)
			std::cout << "sift: saved gss level to " << fm->name << std::endl;

		vl_file_meta_close(fm);
	}

	return VL_ERR_OK;
}

int korder(void const *a, void const *b)
{
	double x = ((double *)a)[2] - ((double *)b)[2];
	if (x < 0) return -1;
	if (x > 0) return +1;
	return 0;
}

bool werr(const vl_bool err, const std::string &name, const std::string &op)
{
    if (VL_ERR_OVERFLOW == err)
	{
		std::cerr << "output file name too long." << std::endl;
		return false;
    }
	else if (err)
	{
		std::cerr << "could not open" << name << "' for " << op << std::endl;
		return false;
    }

	return true;
}

bool qerr(const vl_bool err, const VlFileMeta &ifr)
{
	if (err)
	{
		std::cerr << ifr.name << " malformed" << std::endl;
		return false;
	}

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void sift()
{
	const std::string input_filename = "./machine_vision_data/vlfeat/box.pgm";

	// algorithm parameters
	double edge_thresh = -1;  // edge-thresh
	double peak_thresh = -1;  // peak-thresh
	double magnif = -1;  // magnif
	int O = -1;  // octaves
	int S = 3;  // levels
	int omin = -1;  // first-octave

	VlFileMeta out = { 1, "%.sift",  VL_PROT_ASCII, "", 0 };
	VlFileMeta frm = { 0, "%.frame", VL_PROT_ASCII, "", 0 };
	VlFileMeta dsc = { 0, "%.descr", VL_PROT_ASCII, "", 0 };
	VlFileMeta met = { 0, "%.meta",  VL_PROT_ASCII, "", 0 };
	VlFileMeta gss = { 0, "%.pgm",   VL_PROT_ASCII, "", 0 };
	VlFileMeta ifr = { 0, "%.frame", VL_PROT_ASCII, "", 0 };

	vl_bool err;
    int nikeys = 0, ikeys_size = 0;
    double *ikeys = NULL;

	vl_bool force_output = 0;
	vl_bool force_orientations = 0;
	const bool verbose = true;

    // get basenmae from filename
    char basename[1024];
	{
		const vl_size q = vl_string_basename (basename, sizeof(basename), input_filename.c_str(), 1);

	    err = (q >= sizeof(basename));
	    if (err)
		{
			std::cerr << "basename of '" << input_filename << "' is too long" << std::endl;
			return;
		}
    }

	vl_sift_pix *fdata = NULL;
	VlPgmImage pim;
	if (!local::read_pgm(input_filename, fdata, pim, verbose))
		return;

	// optionally source keypoints
	if (ifr.active)
	{
		// open file
		err = vl_file_meta_open(&ifr, basename, "rb");
		if (!local::werr(err, ifr.name, "reading")) return;

		while (true)
		{
			double x, y, s, th;

			// read next guy
			err = vl_file_meta_get_double(&ifr, &x);
			if (err == VL_ERR_EOF) break;
			else {  if (!local::qerr(err, ifr)) return;  }
			err = vl_file_meta_get_double(&ifr, &y);
			if (!local::qerr(err, ifr)) return;
			err = vl_file_meta_get_double(&ifr, &s);
			if (!local::qerr(err, ifr)) return;
			err = vl_file_meta_get_double(&ifr, &th);
			if (err == VL_ERR_EOF) break;
			else {  if (!local::qerr(err, ifr)) return;  }

			// make enough space
			if (ikeys_size < nikeys + 1)
			{
				ikeys_size += 10000;
				ikeys = new double [4 * ikeys_size];
			}

			// add the guy to the buffer
			ikeys[4 * nikeys + 0] = x;
			ikeys[4 * nikeys + 1] = y;
			ikeys[4 * nikeys + 2] = s;
			ikeys[4 * nikeys + 3] = th;

			++nikeys;
		}

		// now order by scale
		qsort(ikeys, nikeys, 4 * sizeof(double), local::korder);

		if (verbose)
			std::cout << "sift: read " << nikeys << " keypoints from '" << ifr.name << "'" << std::endl;

		// close file
		vl_file_meta_close(&ifr);
	}

	// open output files
	err = vl_file_meta_open(&out, basename, "wb");  if (!local::werr(err, out.name, "writing")) return;
	err = vl_file_meta_open(&dsc, basename, "wb");  if (!local::werr(err, dsc.name, "writing")) return;
	err = vl_file_meta_open(&frm, basename, "wb");  if (!local::werr(err, frm.name, "writing")) return;
	err = vl_file_meta_open(&met, basename, "wb");  if (!local::werr(err, met.name, "writing")) return;

	if (verbose)
	{
		if (out.active) std::cout << "sift: writing all ....... to . " << out.name << std::endl;
		if (frm.active) std::cout << "sift: writing frames .... to . " << frm.name << std::endl;
		if (dsc.active) std::cout << "sift: writing descriptors to . " << dsc.name << std::endl;
		if (met.active) std::cout << "sift: writign meta ...... to . " << met.name << std::endl;
	}

    // make filter
    VlSiftFilt *filt = vl_sift_new(pim.width, pim.height, O, S, omin);

    if (edge_thresh >= 0) vl_sift_set_edge_thresh(filt, edge_thresh);
    if (peak_thresh >= 0) vl_sift_set_peak_thresh(filt, peak_thresh);
    if (magnif >= 0) vl_sift_set_magnif(filt, magnif);

    if (!filt)
	{
		std::cerr << "could not create SIFT filter." << std::endl;
		return;
    }

    if (verbose)
	{
		std::cout << "sift: filter settings:" << std::endl;
		std::cout << "sift:   octaves      (O)     = " << vl_sift_get_noctaves(filt) << std::endl;
		std::cout << "sift:   levels       (S)     = " << vl_sift_get_nlevels(filt) << std::endl;
		std::cout << "sift:   first octave (o_min) = " << vl_sift_get_octave_first(filt) << std::endl;
		std::cout << "sift:   edge thresh          = " << vl_sift_get_edge_thresh(filt) << std::endl;
		std::cout << "sift:   peak thresh          = " << vl_sift_get_peak_thresh(filt) << std::endl;
		std::cout << "sift:   magnif               = " << vl_sift_get_magnif(filt) << std::endl;
		std::cout << "sift: will source frames? " << (ikeys ? "yes" : "no") << std::endl;
		std::cout << "sift: will force orientations? " << (force_orientations ? "yes" : "no") << std::endl;
    }

	// process each octave
	int i = 0;
	vl_bool first = 1;
	while (true)
	{
		VlSiftKeypoint const *keys = 0;
		int nkeys;

		// calculate the GSS for the next octave
		if (first)
		{
			first = 0;
			err = vl_sift_process_first_octave(filt, fdata);
		}
		else
		{
			err = vl_sift_process_next_octave(filt);
		}

		if (err)
		{
			err = VL_ERR_OK;
			break;
		}

		if (verbose)
			std::cout << "sift: GSS octave " << vl_sift_get_octave_index(filt) << " computed" << std::endl;

		// optionally save GSS
		if (gss.active)
		{
			err = local::save_gss(filt, &gss, basename, verbose);
			if (err)
			{
				std::cerr << "could not write GSS to PGM file." << std::endl;
				return;
			}
		}

		// run detector
		if (ikeys == 0)
		{
			vl_sift_detect(filt);

			keys  = vl_sift_get_keypoints(filt);
			nkeys = vl_sift_get_nkeypoints(filt);
			i = 0;

			if (verbose)
				std::cout << "sift: detected " << nkeys << " (unoriented) keypoints" << std::endl;
		}
		else
		{
			nkeys = nikeys;
		}

		// for each keypoint
		for (; i < nkeys; ++i)
		{
			double angles[4];
			int nangles;
			VlSiftKeypoint ik;
			VlSiftKeypoint const *k;

			// obtain keypoint orientations
			if (ikeys)
			{
				vl_sift_keypoint_init(filt, &ik, ikeys [4 * i + 0], ikeys [4 * i + 1], ikeys [4 * i + 2]);

				if (ik.o != vl_sift_get_octave_index(filt)) break;

				k = &ik;

				// optionally compute orientations too
				if (force_orientations)
				{
					nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);
				}
				else
				{
					angles[0] = ikeys[4 * i + 3];
					nangles = 1;
				}
			}
			else
			{
				k = keys + i;
				nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);
			}

			// for each orientation
			for (unsigned q = 0 ; q < (unsigned)nangles ; ++q)
			{
				vl_sift_pix descr[128];

				// compute descriptor (if necessary)
				if (out.active || dsc.active)
				{
					vl_sift_calc_keypoint_descriptor(filt, descr, k, angles [q]);
				}

				if (out.active)
				{
					vl_file_meta_put_double(&out, k->x);
					vl_file_meta_put_double(&out, k->y);
					vl_file_meta_put_double(&out, k->sigma);
					vl_file_meta_put_double(&out, angles[q]);
					for (int l = 0; l < 128; ++l)
					{
						vl_file_meta_put_uint8(&out, (vl_uint8)(512.0 * descr [l]));
					}
					if (out.protocol == VL_PROT_ASCII) fprintf(out.file, "\n");
				}

				if (frm.active)
				{
					vl_file_meta_put_double(&frm, k->x);
					vl_file_meta_put_double(&frm, k->y);
					vl_file_meta_put_double(&frm, k->sigma);
					vl_file_meta_put_double(&frm, angles[q]);
					if (frm.protocol == VL_PROT_ASCII) fprintf(frm.file, "\n");
				}

				if (dsc.active)
				{
					for (int l = 0; l < 128; ++l)
					{
						double x = 512.0 * descr[l];
						x = (x < 255.0) ? x : 255.0;
						vl_file_meta_put_uint8(&dsc, (vl_uint8)(x));
					}
					if (dsc.protocol == VL_PROT_ASCII) fprintf(dsc.file, "\n");
				}
			}
		}
	}

    // finish up
    if (met.active)
	{
		fprintf(met.file, "<sift\n");
		fprintf(met.file, "  input       = '%s'\n", input_filename.c_str());
		if (dsc.active)
			fprintf(met.file, "  descriptors = '%s'\n", dsc.name);
		if (frm.active)
			fprintf(met.file, "  frames      = '%s'\n", frm.name);
		fprintf(met.file, ">\n");
	}

    // release input keys buffer
    if (ikeys)
	{
      delete [] ikeys;
      ikeys_size = nikeys = 0;
      ikeys = NULL;
    }

    // release filter
    if (filt)
	{
      vl_sift_delete(filt);
      filt = NULL;
    }

	// release image data
    if (fdata)
	{
		delete [] fdata;
		fdata = NULL;
	}

    vl_file_meta_close(&out);
    vl_file_meta_close(&frm);
    vl_file_meta_close(&dsc);
    vl_file_meta_close(&met);
    vl_file_meta_close(&gss);
    vl_file_meta_close(&ifr);
}

}  // namespace my_vlfeat
