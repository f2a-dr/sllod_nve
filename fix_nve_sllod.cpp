// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_nve_sllod.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_deform.h"
#include "force.h"
#include "group.h"
#include "math_extra.h"
#include "modify.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVESllod::FixNVESllod(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (!utils::strmatch(style,"^nve/sphere") && narg < 3)
    utils::missing_cmd_args(FLERR, "fix nve", error);

  dynamic_group_allow = 1;
  time_integrate = 1;

  // default value for psllod
 
  psllod_flag = 0;

  // select SLLOD/p-SLLOD/g-SLLOD variant

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "psllod") == 0) {
      if (iarg+2 > narg) utils::missing_cmd_args(FLERR, "fix nve/sllod psllod", error);
      psllod_flag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else iarg++;
  }
  
  // Addition to compute the bias

  id_temp = utils::strdup(std::string(id) + "_temp");
  modify->add_compute(fmt::format("{} {} temp/deform",id_temp,group->names[igroup]));
  tcomputeflag = 1;
  nondeformbias = 0;

  // set temperature and pressure ptrs

  temperature = modify->get_compute_by_id(id_temp);
  if (!temperature) error->all(FLERR,"Temperature ID {} for fix {} does not exist", id_temp, style);
}

/* ---------------------------------------------------------------------- */

FixNVESllod::~FixNVESllod()
{
  // delete temperature and pressure if fix created them

  if (tcomputeflag) modify->delete_compute(id_temp);
  delete[] id_temp;
}

/* ---------------------------------------------------------------------- */

int FixNVESllod::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVESllod::init()
{
  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix {} does not have a bias", style);

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform") != 0) nondeformbias = 1;

  // check fix deform remap settings

  auto deform = modify->get_fix_by_style("^deform");
  if (deform.size() < 1) error->all(FLERR,"Using fix {} with no fix deform defined", style);

  for (auto &ifix : deform) {
    auto f = dynamic_cast<FixDeform *>(ifix);
    if (f && (f->remapflag != Domain::V_REMAP))
      error->all(FLERR,"Using fix {} with inconsistent fix deform remap option", style);
  }

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dthalf = 0.5 * update->dt;

  if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVESllod::initial_integrate(int /*vflag*/)
{
  double dtfm;

  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //    calculate temperature since some computes require temp
  //    computed on current nlocal atoms to remove bias

  if (nondeformbias) temperature->compute_scalar();

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        if (!psllod_flag) temperature->remove_bias(i,v[i]);
        vdelu[0] = h_two[0]*v[i][0] + h_two[5]*v[i][1] + h_two[4]*v[i][2];
        vdelu[1] = h_two[1]*v[i][1] + h_two[3]*v[i][2];
        vdelu[2] = h_two[2]*v[i][2];
        if (psllod_flag) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0] - dthalf*vdelu[0];
        v[i][1] = v[i][1] - dthalf*vdelu[1];
        v[i][2] = v[i][2] - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        if (!psllod_flag) temperature->remove_bias(i,v[i]);
        vdelu[0] = h_two[0]*v[i][0] + h_two[5]*v[i][1] + h_two[4]*v[i][2];
        vdelu[1] = h_two[1]*v[i][1] + h_two[3]*v[i][2];
        vdelu[2] = h_two[2]*v[i][2];
        if (psllod_flag) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0] - dthalf*vdelu[0];
        v[i][1] = v[i][1] - dthalf*vdelu[1];
        v[i][2] = v[i][2] - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        x[i][0] += dtv * v[i][0];
        x[i][1] += dtv * v[i][1];
        x[i][2] += dtv * v[i][2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESllod::final_integrate()
{
  double dtfm;

  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //    calculate temperature since some computes require temp
  //    computed on current nlocal atoms to remove bias

  if (nondeformbias) temperature->compute_scalar();

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double h_two[6],vdelu[3];
  MathExtra::multiply_shape_shape(domain->h_rate,domain->h_inv,h_two);

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        if (!psllod_flag) temperature->remove_bias(i,v[i]);
        vdelu[0] = h_two[0]*v[i][0] + h_two[5]*v[i][1] + h_two[4]*v[i][2];
        vdelu[1] = h_two[1]*v[i][1] + h_two[3]*v[i][2];
        vdelu[2] = h_two[2]*v[i][2];
        if (psllod_flag) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0] - dthalf*vdelu[0];
        v[i][1] = v[i][1] - dthalf*vdelu[1];
        v[i][2] = v[i][2] - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        if (!psllod_flag) temperature->remove_bias(i,v[i]);
        vdelu[0] = h_two[0]*v[i][0] + h_two[5]*v[i][1] + h_two[4]*v[i][2];
        vdelu[1] = h_two[1]*v[i][1] + h_two[3]*v[i][2];
        vdelu[2] = h_two[2]*v[i][2];
        if (psllod_flag) temperature->remove_bias(i,v[i]);
        v[i][0] = v[i][0] - dthalf*vdelu[0];
        v[i][1] = v[i][1] - dthalf*vdelu[1];
        v[i][2] = v[i][2] - dthalf*vdelu[2];
        temperature->restore_bias(i,v[i]);
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVESllod::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESllod::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVESllod::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
