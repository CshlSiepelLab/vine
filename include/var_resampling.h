/* PHylogenetic Analysis with Space/Time models
 * Copyright (c) 2002-2005 University of California, 2006-2010 Cornell 
 * University.  All rights reserved.
 *
 * This source code is distributed under a BSD-style license.  See the
 * file LICENSE.txt for details.
 ***************************************************************************/

/** @file upgma.h
    Simple UPGMA tree inference  
    @ingroup phylo
*/

#ifndef VARRES_H
#define VARRES_H

#include <stdio.h>
#include <limits.h>
#include <phast/tree_model.h>
#include <mvn.h>
#include <multi_mvn.h>
#include <nj.h>

List *nj_importance_sample(int nsamples, List *trees, Vector *logdens,
                           TreeModel *mod, CovarData *data, FILE *logf);

List *nj_var_sample_rejection(int nsamples, multi_MVN *mmvn,
                              CovarData *data, TreeModel *mod,
                              FILE *logf);

List *nj_var_sample_importance(int nsamples, multi_MVN *mmvn,
                               CovarData *data, TreeModel *mod,
                               FILE *logf);

#endif
