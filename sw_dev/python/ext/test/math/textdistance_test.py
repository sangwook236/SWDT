#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import textdistance

# REF [site] >> https://github.com/life4/textdistance
def simple_example():
	str1, str2 = 'test', 'text'
	qval = 2

	#--------------------
	# Edit-based.
	if True:
		print("textdistance.hamming({}, {}) = {}.".format(str1, str2, textdistance.hamming(str1, str2)))
		print("textdistance.hamming.distance({}, {}) = {}.".format(str1, str2, textdistance.hamming.distance(str1, str2)))
		print("textdistance.hamming.similarity({}, {}) = {}.".format(str1, str2, textdistance.hamming.similarity(str1, str2)))
		print("textdistance.hamming.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.hamming.normalized_distance(str1, str2)))
		print("textdistance.hamming.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.hamming.normalized_similarity(str1, str2)))
		print("textdistance.Hamming(qval={}, test_func=None, truncate=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Hamming(qval=qval, test_func=None, truncate=False, external=True).distance(str1, str2)))

		print("textdistance.mlipns({}, {}) = {}.".format(str1, str2, textdistance.mlipns(str1, str2)))
		print("textdistance.mlipns.distance({}, {}) = {}.".format(str1, str2, textdistance.mlipns.distance(str1, str2)))
		print("textdistance.mlipns.similarity({}, {}) = {}.".format(str1, str2, textdistance.mlipns.similarity(str1, str2)))
		print("textdistance.mlipns.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.mlipns.normalized_distance(str1, str2)))
		print("textdistance.mlipns.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.mlipns.normalized_similarity(str1, str2)))
		print("textdistance.MLIPNS(threshold=0.25, maxmismatches=2, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.MLIPNS(threshold=0.25, maxmismatches=2, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.levenshtein({}, {}) = {}.".format(str1, str2, textdistance.levenshtein(str1, str2)))
		print("textdistance.levenshtein.distance({}, {}) = {}.".format(str1, str2, textdistance.levenshtein.distance(str1, str2)))
		print("textdistance.levenshtein.similarity({}, {}) = {}.".format(str1, str2, textdistance.levenshtein.similarity(str1, str2)))
		print("textdistance.levenshtein.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.levenshtein.normalized_distance(str1, str2)))
		print("textdistance.levenshtein.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.levenshtein.normalized_similarity(str1, str2)))
		print("textdistance.Levenshtein(qval={}, test_func=None, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Levenshtein(qval=qval, test_func=None, external=True).distance(str1, str2)))

		print("textdistance.damerau_levenshtein({}, {}) = {}.".format(str1, str2, textdistance.damerau_levenshtein(str1, str2)))
		print("textdistance.damerau_levenshtein.distance({}, {}) = {}.".format(str1, str2, textdistance.damerau_levenshtein.distance(str1, str2)))
		print("textdistance.damerau_levenshtein.similarity({}, {}) = {}.".format(str1, str2, textdistance.damerau_levenshtein.similarity(str1, str2)))
		print("textdistance.damerau_levenshtein.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.damerau_levenshtein.normalized_distance(str1, str2)))
		print("textdistance.damerau_levenshtein.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.damerau_levenshtein.normalized_similarity(str1, str2)))
		print("textdistance.DamerauLevenshtein(qval={}, test_func=None, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.DamerauLevenshtein(qval=qval, test_func=None, external=True).distance(str1, str2)))

		print("textdistance.jaro({}, {}) = {}.".format(str1, str2, textdistance.jaro(str1, str2)))
		print("textdistance.jaro.distance({}, {}) = {}.".format(str1, str2, textdistance.jaro.distance(str1, str2)))
		print("textdistance.jaro.similarity({}, {}) = {}.".format(str1, str2, textdistance.jaro.similarity(str1, str2)))
		print("textdistance.jaro.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.jaro.normalized_distance(str1, str2)))
		print("textdistance.jaro.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.jaro.normalized_similarity(str1, str2)))
		print("textdistance.Jaro(long_tolerance=False, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Jaro(long_tolerance=False, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.jaro_winkler({}, {}) = {}.".format(str1, str2, textdistance.jaro_winkler(str1, str2)))
		print("textdistance.jaro_winkler.distance({}, {}) = {}.".format(str1, str2, textdistance.jaro_winkler.distance(str1, str2)))
		print("textdistance.jaro_winkler.similarity({}, {}) = {}.".format(str1, str2, textdistance.jaro_winkler.similarity(str1, str2)))
		print("textdistance.jaro_winkler.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.jaro_winkler.normalized_distance(str1, str2)))
		print("textdistance.jaro_winkler.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.jaro_winkler.normalized_similarity(str1, str2)))
		print("textdistance.JaroWinkler(long_tolerance=False, winklerize=True, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.JaroWinkler(long_tolerance=False, winklerize=True, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.strcmp95({}, {}) = {}.".format(str1, str2, textdistance.strcmp95(str1, str2)))
		print("textdistance.strcmp95.distance({}, {}) = {}.".format(str1, str2, textdistance.strcmp95.distance(str1, str2)))
		print("textdistance.strcmp95.similarity({}, {}) = {}.".format(str1, str2, textdistance.strcmp95.similarity(str1, str2)))
		print("textdistance.strcmp95.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.strcmp95.normalized_distance(str1, str2)))
		print("textdistance.strcmp95.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.strcmp95.normalized_similarity(str1, str2)))
		print("textdistance.StrCmp95(long_strings=False, external=True).distance({}, {}) = {}.".format(str1, str2, textdistance.StrCmp95(long_strings=False, external=True).distance(str1, str2)))

		print("textdistance.needleman_wunsch({}, {}) = {}.".format(str1, str2, textdistance.needleman_wunsch(str1, str2)))
		print("textdistance.needleman_wunsch.distance({}, {}) = {}.".format(str1, str2, textdistance.needleman_wunsch.distance(str1, str2)))
		print("textdistance.needleman_wunsch.similarity({}, {}) = {}.".format(str1, str2, textdistance.needleman_wunsch.similarity(str1, str2)))
		print("textdistance.needleman_wunsch.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.needleman_wunsch.normalized_distance(str1, str2)))
		print("textdistance.needleman_wunsch.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.needleman_wunsch.normalized_similarity(str1, str2)))
		print("textdistance.NeedlemanWunsch(gap_cost=1.0, sim_func=None, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.NeedlemanWunsch(gap_cost=1.0, sim_func=None, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.gotoh({}, {}) = {}.".format(str1, str2, textdistance.gotoh(str1, str2)))
		print("textdistance.gotoh.distance({}, {}) = {}.".format(str1, str2, textdistance.gotoh.distance(str1, str2)))
		print("textdistance.gotoh.similarity({}, {}) = {}.".format(str1, str2, textdistance.gotoh.similarity(str1, str2)))
		print("textdistance.gotoh.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.gotoh.normalized_distance(str1, str2)))
		print("textdistance.gotoh.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.gotoh.normalized_similarity(str1, str2)))
		print("textdistance.Gotoh(gap_open=1, gap_ext=0.4, sim_func=None, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Gotoh(gap_open=1, gap_ext=0.4, sim_func=None, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.smith_waterman({}, {}) = {}.".format(str1, str2, textdistance.smith_waterman(str1, str2)))
		print("textdistance.smith_waterman.distance({}, {}) = {}.".format(str1, str2, textdistance.smith_waterman.distance(str1, str2)))
		print("textdistance.smith_waterman.similarity({}, {}) = {}.".format(str1, str2, textdistance.smith_waterman.similarity(str1, str2)))
		print("textdistance.smith_waterman.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.smith_waterman.normalized_distance(str1, str2)))
		print("textdistance.smith_waterman.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.smith_waterman.normalized_similarity(str1, str2)))
		print("textdistance.SmithWaterman(gap_cost=1.0, sim_func=None, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.SmithWaterman(gap_cost=1.0, sim_func=None, qval=qval, external=True).distance(str1, str2)))

	#--------------------
	# Token-based.
	if False:
		print("textdistance.jaccard({}, {}) = {}.".format(str1, str2, textdistance.jaccard(str1, str2)))
		print("textdistance.jaccard.distance({}, {}) = {}.".format(str1, str2, textdistance.jaccard.distance(str1, str2)))
		print("textdistance.jaccard.similarity({}, {}) = {}.".format(str1, str2, textdistance.jaccard.similarity(str1, str2)))
		print("textdistance.jaccard.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.jaccard.normalized_distance(str1, str2)))
		print("textdistance.jaccard.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.jaccard.normalized_similarity(str1, str2)))
		print("textdistance.Jaccard(qval={}, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Jaccard(qval=qval, as_set=False, external=True).distance(str1, str2)))

		print("textdistance.sorensen({}, {}) = {}.".format(str1, str2, textdistance.sorensen(str1, str2)))
		print("textdistance.sorensen.distance({}, {}) = {}.".format(str1, str2, textdistance.sorensen.distance(str1, str2)))
		print("textdistance.sorensen.similarity({}, {}) = {}.".format(str1, str2, textdistance.sorensen.similarity(str1, str2)))
		print("textdistance.sorensen.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.sorensen.normalized_distance(str1, str2)))
		print("textdistance.sorensen.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.sorensen.normalized_similarity(str1, str2)))
		print("textdistance.Sorensen(qval={}, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Sorensen(qval=qval, as_set=False, external=True).distance(str1, str2)))

		print("textdistance.sorensen_dice({}, {}) = {}.".format(str1, str2, textdistance.sorensen_dice(str1, str2)))
		print("textdistance.sorensen_dice.distance({}, {}) = {}.".format(str1, str2, textdistance.sorensen_dice.distance(str1, str2)))
		print("textdistance.sorensen_dice.similarity({}, {}) = {}.".format(str1, str2, textdistance.sorensen_dice.similarity(str1, str2)))
		print("textdistance.sorensen_dice.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.sorensen_dice.normalized_distance(str1, str2)))
		print("textdistance.sorensen_dice.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.sorensen_dice.normalized_similarity(str1, str2)))
		#print("textdistance.SorensenDice().distance({}, {}) = {}.".format(str1, str2, textdistance.SorensenDice().distance(str1, str2)))

		print("textdistance.tversky({}, {}) = {}.".format(str1, str2, textdistance.tversky(str1, str2)))
		print("textdistance.tversky.distance({}, {}) = {}.".format(str1, str2, textdistance.tversky.distance(str1, str2)))
		print("textdistance.tversky.similarity({}, {}) = {}.".format(str1, str2, textdistance.tversky.similarity(str1, str2)))
		print("textdistance.tversky.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.tversky.normalized_distance(str1, str2)))
		print("textdistance.tversky.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.tversky.normalized_similarity(str1, str2)))
		print("textdistance.Tversky(qval={}, ks=None, bias=None, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Tversky(qval=qval, ks=None, bias=None, as_set=False, external=True).distance(str1, str2)))

		print("textdistance.overlap({}, {}) = {}.".format(str1, str2, textdistance.overlap(str1, str2)))
		print("textdistance.overlap.distance({}, {}) = {}.".format(str1, str2, textdistance.overlap.distance(str1, str2)))
		print("textdistance.overlap.similarity({}, {}) = {}.".format(str1, str2, textdistance.overlap.similarity(str1, str2)))
		print("textdistance.overlap.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.overlap.normalized_distance(str1, str2)))
		print("textdistance.overlap.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.overlap.normalized_similarity(str1, str2)))
		print("textdistance.Overlap(qval={}, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Overlap(qval=qval, as_set=False, external=True).distance(str1, str2)))

		# This is identical to the Jaccard similarity coefficient and the Tversky index for alpha=1 and beta=1.
		print("textdistance.tanimoto({}, {}) = {}.".format(str1, str2, textdistance.tanimoto(str1, str2)))
		print("textdistance.tanimoto.distance({}, {}) = {}.".format(str1, str2, textdistance.tanimoto.distance(str1, str2)))
		print("textdistance.tanimoto.similarity({}, {}) = {}.".format(str1, str2, textdistance.tanimoto.similarity(str1, str2)))
		print("textdistance.tanimoto.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.tanimoto.normalized_distance(str1, str2)))
		print("textdistance.tanimoto.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.tanimoto.normalized_similarity(str1, str2)))
		print("textdistance.Tanimoto(qval={}, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Tanimoto(qval=qval, as_set=False, external=True).distance(str1, str2)))

		print("textdistance.cosine({}, {}) = {}.".format(str1, str2, textdistance.cosine(str1, str2)))
		print("textdistance.cosine.distance({}, {}) = {}.".format(str1, str2, textdistance.cosine.distance(str1, str2)))
		print("textdistance.cosine.similarity({}, {}) = {}.".format(str1, str2, textdistance.cosine.similarity(str1, str2)))
		print("textdistance.cosine.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.cosine.normalized_distance(str1, str2)))
		print("textdistance.cosine.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.cosine.normalized_similarity(str1, str2)))
		print("textdistance.Cosine(qval={}, as_set=False, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Cosine(qval=qval, as_set=False, external=True).distance(str1, str2)))

		print("textdistance.monge_elkan({}, {}) = {}.".format(str1, str2, textdistance.monge_elkan(str1, str2)))
		print("textdistance.monge_elkan.distance({}, {}) = {}.".format(str1, str2, textdistance.monge_elkan.distance(str1, str2)))
		print("textdistance.monge_elkan.similarity({}, {}) = {}.".format(str1, str2, textdistance.monge_elkan.similarity(str1, str2)))
		print("textdistance.monge_elkan.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.monge_elkan.normalized_distance(str1, str2)))
		print("textdistance.monge_elkan.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.monge_elkan.normalized_similarity(str1, str2)))
		print("textdistance.MongeElkan(algorithm=textdistance.DamerauLevenshtein(), symmetric=False, qval={}, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.MongeElkan(algorithm=textdistance.DamerauLevenshtein(), symmetric=False, qval=qval, external=True).distance(str1, str2)))

		print("textdistance.bag({}, {}) = {}.".format(str1, str2, textdistance.bag(str1, str2)))
		print("textdistance.bag.distance({}, {}) = {}.".format(str1, str2, textdistance.bag.distance(str1, str2)))
		print("textdistance.bag.similarity({}, {}) = {}.".format(str1, str2, textdistance.bag.similarity(str1, str2)))
		print("textdistance.bag.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.bag.normalized_distance(str1, str2)))
		print("textdistance.bag.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.bag.normalized_similarity(str1, str2)))
		print("textdistance.Bag(qval={}).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Bag(qval=qval).distance(str1, str2)))

	#--------------------
	# Sequence-based.
	if False:
		print("textdistance.lcsseq({}, {}) = {}.".format(str1, str2, textdistance.lcsseq(str1, str2)))
		print("textdistance.lcsseq.distance({}, {}) = {}.".format(str1, str2, textdistance.lcsseq.distance(str1, str2)))
		print("textdistance.lcsseq.similarity({}, {}) = {}.".format(str1, str2, textdistance.lcsseq.similarity(str1, str2)))
		print("textdistance.lcsseq.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.lcsseq.normalized_distance(str1, str2)))
		print("textdistance.lcsseq.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.lcsseq.normalized_similarity(str1, str2)))
		#print("textdistance.LCSSeq(qval={}, test_func=None, external=True).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.LCSSeq(qval=qval, test_func=None, external=True).distance(str1, str2)))
		print("textdistance.LCSSeq().distance({}, {}) = {}.".format(str1, str2, textdistance.LCSSeq().distance(str1, str2)))

		print("textdistance.lcsstr({}, {}) = {}.".format(str1, str2, textdistance.lcsstr(str1, str2)))
		print("textdistance.lcsstr.distance({}, {}) = {}.".format(str1, str2, textdistance.lcsstr.distance(str1, str2)))
		print("textdistance.lcsstr.similarity({}, {}) = {}.".format(str1, str2, textdistance.lcsstr.similarity(str1, str2)))
		print("textdistance.lcsstr.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.lcsstr.normalized_distance(str1, str2)))
		print("textdistance.lcsstr.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.lcsstr.normalized_similarity(str1, str2)))
		print("textdistance.LCSStr(qval={}).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.LCSStr(qval=qval).distance(str1, str2)))

		print("textdistance.ratcliff_obershelp({}, {}) = {}.".format(str1, str2, textdistance.ratcliff_obershelp(str1, str2)))
		print("textdistance.ratcliff_obershelp.distance({}, {}) = {}.".format(str1, str2, textdistance.ratcliff_obershelp.distance(str1, str2)))
		print("textdistance.ratcliff_obershelp.similarity({}, {}) = {}.".format(str1, str2, textdistance.ratcliff_obershelp.similarity(str1, str2)))
		print("textdistance.ratcliff_obershelp.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.ratcliff_obershelp.normalized_distance(str1, str2)))
		print("textdistance.ratcliff_obershelp.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.ratcliff_obershelp.normalized_similarity(str1, str2)))
		print("textdistance.RatcliffObershelp().distance({}, {}) = {}.".format(str1, str2, textdistance.RatcliffObershelp().distance(str1, str2)))

	#--------------------
	# Compression-based.
	if False:
		print("textdistance.arith_ncd({}, {}) = {}.".format(str1, str2, textdistance.arith_ncd(str1, str2)))
		print("textdistance.arith_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.arith_ncd.distance(str1, str2)))
		print("textdistance.arith_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.arith_ncd.similarity(str1, str2)))
		print("textdistance.arith_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.arith_ncd.normalized_distance(str1, str2)))
		print("textdistance.arith_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.arith_ncd.normalized_similarity(str1, str2)))
		#print("textdistance.ArithNCD(base=2, terminator=None, qval={}).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.ArithNCD(base=2, terminator=None, qval=qval).distance(str1, str2)))
		print("textdistance.ArithNCD().distance({}, {}) = {}.".format(str1, str2, textdistance.ArithNCD().distance(str1, str2)))

		print("textdistance.rle_ncd({}, {}) = {}.".format(str1, str2, textdistance.rle_ncd(str1, str2)))
		print("textdistance.rle_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.rle_ncd.distance(str1, str2)))
		print("textdistance.rle_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.rle_ncd.similarity(str1, str2)))
		print("textdistance.rle_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.rle_ncd.normalized_distance(str1, str2)))
		print("textdistance.rle_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.rle_ncd.normalized_similarity(str1, str2)))
		print("textdistance.RLENCD().distance({}, {}) = {}.".format(str1, str2, textdistance.RLENCD().distance(str1, str2)))

		print("textdistance.bwtrle_ncd({}, {}) = {}.".format(str1, str2, textdistance.bwtrle_ncd(str1, str2)))
		print("textdistance.bwtrle_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.bwtrle_ncd.distance(str1, str2)))
		print("textdistance.bwtrle_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.bwtrle_ncd.similarity(str1, str2)))
		print("textdistance.bwtrle_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.bwtrle_ncd.normalized_distance(str1, str2)))
		print("textdistance.bwtrle_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.bwtrle_ncd.normalized_similarity(str1, str2)))
		print("textdistance.BWTRLENCD(terminator='\0').distance({}, {}) = {}.".format(str1, str2, textdistance.BWTRLENCD(terminator='\0').distance(str1, str2)))

		print("textdistance.sqrt_ncd({}, {}) = {}.".format(str1, str2, textdistance.sqrt_ncd(str1, str2)))
		print("textdistance.sqrt_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.sqrt_ncd.distance(str1, str2)))
		print("textdistance.sqrt_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.sqrt_ncd.similarity(str1, str2)))
		print("textdistance.sqrt_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.sqrt_ncd.normalized_distance(str1, str2)))
		print("textdistance.sqrt_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.sqrt_ncd.normalized_similarity(str1, str2)))
		print("textdistance.SqrtNCD(qval={}).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.SqrtNCD(qval=qval).distance(str1, str2)))

		print("textdistance.entropy_ncd({}, {}) = {}.".format(str1, str2, textdistance.entropy_ncd(str1, str2)))
		print("textdistance.entropy_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.entropy_ncd.distance(str1, str2)))
		print("textdistance.entropy_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.entropy_ncd.similarity(str1, str2)))
		print("textdistance.entropy_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.entropy_ncd.normalized_distance(str1, str2)))
		print("textdistance.entropy_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.entropy_ncd.normalized_similarity(str1, str2)))
		print("textdistance.EntropyNCD(qval={}, coef=1, base=2).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.EntropyNCD(qval=qval, coef=1, base=2).distance(str1, str2)))

		print("textdistance.bz2_ncd({}, {}) = {}.".format(str1, str2, textdistance.bz2_ncd(str1, str2)))
		print("textdistance.bz2_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.bz2_ncd.distance(str1, str2)))
		print("textdistance.bz2_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.bz2_ncd.similarity(str1, str2)))
		print("textdistance.bz2_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.bz2_ncd.normalized_distance(str1, str2)))
		print("textdistance.bz2_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.bz2_ncd.normalized_similarity(str1, str2)))
		print("textdistance.BZ2NCD().distance({}, {}) = {}.".format(str1, str2, textdistance.BZ2NCD().distance(str1, str2)))

		print("textdistance.lzma_ncd({}, {}) = {}.".format(str1, str2, textdistance.lzma_ncd(str1, str2)))
		print("textdistance.lzma_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.lzma_ncd.distance(str1, str2)))
		print("textdistance.lzma_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.lzma_ncd.similarity(str1, str2)))
		print("textdistance.lzma_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.lzma_ncd.normalized_distance(str1, str2)))
		print("textdistance.lzma_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.lzma_ncd.normalized_similarity(str1, str2)))
		print("textdistance.LZMANCD().distance({}, {}) = {}.".format(str1, str2, textdistance.LZMANCD().distance(str1, str2)))

		print("textdistance.zlib_ncd({}, {}) = {}.".format(str1, str2, textdistance.zlib_ncd(str1, str2)))
		print("textdistance.zlib_ncd.distance({}, {}) = {}.".format(str1, str2, textdistance.zlib_ncd.distance(str1, str2)))
		print("textdistance.zlib_ncd.similarity({}, {}) = {}.".format(str1, str2, textdistance.zlib_ncd.similarity(str1, str2)))
		print("textdistance.zlib_ncd.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.zlib_ncd.normalized_distance(str1, str2)))
		print("textdistance.zlib_ncd.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.zlib_ncd.normalized_similarity(str1, str2)))
		print("textdistance.ZLIBNCD().distance({}, {}) = {}.".format(str1, str2, textdistance.ZLIBNCD().distance(str1, str2)))

	#--------------------
	# Phonetic.
	if False:
		print("textdistance.mra({}, {}) = {}.".format(str1, str2, textdistance.mra(str1, str2)))
		print("textdistance.mra.distance({}, {}) = {}.".format(str1, str2, textdistance.mra.distance(str1, str2)))
		print("textdistance.mra.similarity({}, {}) = {}.".format(str1, str2, textdistance.mra.similarity(str1, str2)))
		print("textdistance.mra.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.mra.normalized_distance(str1, str2)))
		print("textdistance.mra.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.mra.normalized_similarity(str1, str2)))
		print("textdistance.MRA().distance({}, {}) = {}.".format(str1, str2, textdistance.MRA().distance(str1, str2)))

		print("textdistance.editex({}, {}) = {}.".format(str1, str2, textdistance.editex(str1, str2)))
		print("textdistance.editex.distance({}, {}) = {}.".format(str1, str2, textdistance.editex.distance(str1, str2)))
		print("textdistance.editex.similarity({}, {}) = {}.".format(str1, str2, textdistance.editex.similarity(str1, str2)))
		print("textdistance.editex.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.editex.normalized_distance(str1, str2)))
		print("textdistance.editex.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.editex.normalized_similarity(str1, str2)))
		print("textdistance.Editex(local=False, match_cost=0, group_cost=1, mismatch_cost=2, groups=None, ungrouped=None, external=True).distance({}, {}) = {}.".format(str1, str2, textdistance.Editex(local=False, match_cost=0, group_cost=1, mismatch_cost=2, groups=None, ungrouped=None, external=True).distance(str1, str2)))

	#--------------------
	# Simple.
	if False:
		print("textdistance.prefix({}, {}) = {}.".format(str1, str2, textdistance.prefix(str1, str2)))
		print("textdistance.prefix.distance({}, {}) = {}.".format(str1, str2, textdistance.prefix.distance(str1, str2)))
		print("textdistance.prefix.similarity({}, {}) = {}.".format(str1, str2, textdistance.prefix.similarity(str1, str2)))
		print("textdistance.prefix.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.prefix.normalized_distance(str1, str2)))
		print("textdistance.prefix.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.prefix.normalized_similarity(str1, str2)))
		print("textdistance.Prefix(qval={}, sim_test=None).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Prefix(qval=qval, sim_test=None).distance(str1, str2)))

		print("textdistance.postfix({}, {}) = {}.".format(str1, str2, textdistance.postfix(str1, str2)))
		print("textdistance.postfix.distance({}, {}) = {}.".format(str1, str2, textdistance.postfix.distance(str1, str2)))
		print("textdistance.postfix.similarity({}, {}) = {}.".format(str1, str2, textdistance.postfix.similarity(str1, str2)))
		print("textdistance.postfix.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.postfix.normalized_distance(str1, str2)))
		print("textdistance.postfix.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.postfix.normalized_similarity(str1, str2)))
		#print("textdistance.Postfix(qval={}, sim_test=None).distance({}, {}) = {}.".format(qval, str1, str2, textdistance.Postfix(qval=qval, sim_test=None).distance(str1, str2)))
		print("textdistance.Postfix().distance({}, {}) = {}.".format(str1, str2, textdistance.Postfix().distance(str1, str2)))

		print("textdistance.length({}, {}) = {}.".format(str1, str2, textdistance.length(str1, str2)))
		print("textdistance.length.distance({}, {}) = {}.".format(str1, str2, textdistance.length.distance(str1, str2)))
		print("textdistance.length.similarity({}, {}) = {}.".format(str1, str2, textdistance.length.similarity(str1, str2)))
		print("textdistance.length.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.length.normalized_distance(str1, str2)))
		print("textdistance.length.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.length.normalized_similarity(str1, str2)))
		print("textdistance.Length().distance({}, {}) = {}.".format(str1, str2, textdistance.Length().distance(str1, str2)))

		print("textdistance.identity({}, {}) = {}.".format(str1, str2, textdistance.identity(str1, str2)))
		print("textdistance.identity.distance({}, {}) = {}.".format(str1, str2, textdistance.identity.distance(str1, str2)))
		print("textdistance.identity.similarity({}, {}) = {}.".format(str1, str2, textdistance.identity.similarity(str1, str2)))
		print("textdistance.identity.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.identity.normalized_distance(str1, str2)))
		print("textdistance.identity.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.identity.normalized_similarity(str1, str2)))
		print("textdistance.Identity().distance({}, {}) = {}.".format(str1, str2, textdistance.Identity().distance(str1, str2)))

		print("textdistance.matrix({}, {}) = {}.".format(str1, str2, textdistance.matrix(str1, str2)))
		print("textdistance.matrix.distance({}, {}) = {}.".format(str1, str2, textdistance.matrix.distance(str1, str2)))
		print("textdistance.matrix.similarity({}, {}) = {}.".format(str1, str2, textdistance.matrix.similarity(str1, str2)))
		print("textdistance.matrix.normalized_distance({}, {}) = {}.".format(str1, str2, textdistance.matrix.normalized_distance(str1, str2)))
		print("textdistance.matrix.normalized_similarity({}, {}) = {}.".format(str1, str2, textdistance.matrix.normalized_similarity(str1, str2)))
		print("textdistance.Matrix(mat=None, mismatch_cost=0, match_cost=1, symmetric=True, external=True).distance({}, {}) = {}.".format(str1, str2, textdistance.Matrix(mat=None, mismatch_cost=0, match_cost=1, symmetric=True, external=True).distance(str1, str2)))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
