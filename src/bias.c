/** @file bias.c
 *  @brief Halo bias functions.
 * 
 *  These functions are the halo bias.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_bias.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define delta_c 1.686 //Critical collapse density

/**
 * \brief Compute the bias of a halo with peak height nu for an array
 * of peak heights, with arbitrary free parameters in a Tinker-like model.
 *
 */
int bias_at_nu_arr_FREEPARAMS(double*nu, int Nnu, int delta,
			      double A, double a, double B, double b,
			      double C, double c, double*bias){
  int i;
  for(i = 0; i < Nnu; i++)
    bias[i] = 1 - A*pow(nu[i],a)/(pow(nu[i],a)+pow(delta_c,a)) + B*pow(nu[i],b) + C*pow(nu[i],c);
  return 0;
}
