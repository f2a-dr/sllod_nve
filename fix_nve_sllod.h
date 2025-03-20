/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/sllod,FixNVESllod);
// clang-format on
#else

#ifndef LMP_FIX_NVE_SLLOD_H
#define LMP_FIX_NVE_SLLOD_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNVESllod : public Fix {
 public:
  FixNVESllod(class LAMMPS *, int, char **);
  ~FixNVESllod() override;
  int setmask() override;
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void initial_integrate_respa(int, int, int) override;
  void final_integrate_respa(int, int) override;
  void reset_dt() override;

 protected:
  double dtv, dtf, dthalf;
  double *step_respa;
  int mass_require;

  char *id_temp;
  class Compute *temperature;
  int tcomputeflag;     // 1 = compute was created by fix
                        // 0 = created exertnally

 private:
  int nondeformbias;
  int psllod_flag;     // 0 for SLLOD, 1 for p-SLLOD
};

}    // namespace LAMMPS_NS

#endif
#endif
