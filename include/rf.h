/*
 * VINE: Variational Inference with Node Embeddings
 *
 * Copyright (c) 2025, Adam Siepel
 * All rights reserved.
 *
 * This file is part of VINE and is distributed under the BSD 3-Clause License.
 * See the LICENSE file in the project root for details.
 */

/* calculation of Robinson Foulds distances */

#ifndef RF_H
#define RF_H

#include <stdio.h>

/* bitset for up to many thousands of leaves */
typedef struct {
  int W;            /* number of 64-bit words */
  uint64_t *w;      /* words */
} BitMask;

/* dynamic array of BitMask* */
typedef struct {
  BitMask **a; int size, cap;
} MaskVec;

double tr_robinson_foulds(TreeNode *t1, TreeNode *t2);

#endif
