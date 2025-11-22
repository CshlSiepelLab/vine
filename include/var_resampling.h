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
