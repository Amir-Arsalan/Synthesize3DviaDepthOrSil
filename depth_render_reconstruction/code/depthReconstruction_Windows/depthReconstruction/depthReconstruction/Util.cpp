//============================================================================
//
// This file is part of the Thea project.
//
// This software is covered by the following BSD license, except for portions
// derived from other works which are covered by their respective licenses.
// For full licensing information including reproduction of these external
// licenses, see the file LICENSE.txt provided in the documentation.
//
// Copyright (c) 2011, Stanford University
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holders nor the names of contributors
// to this software may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//============================================================================

#include "Util.hpp"

namespace Thea {

namespace Triangle3Internal {

Real
closestPtSegmentSegment(Vector3 const & p1, Vector3 const & q1, Vector3 const & p2, Vector3 const & q2, Real & s, Real & t,
                        Vector3 & c1, Vector3 & c2)
{
  static Real const EPSILON = 1e-10f;

  Vector3 d1 = q1 - p1;  // Direction vector of segment S1
  Vector3 d2 = q2 - p2;  // Direction vector of segment S2
  Vector3 r = p1 - p2;
  Real a = d1.squaredLength();  // Squared length of segment S1, always nonnegative
  Real e = d2.squaredLength();  // Squared length of segment S2, always nonnegative
  Real f = d2.dot(r);

  // Check if either or both segments degenerate into points
  if (a <= EPSILON && e <= EPSILON)
  {
    // Both segments degenerate into points
    s = t = 0;
    c1 = p1;
    c2 = p2;
    return (c1 - c2).dot(c1 - c2);
  }
  if (a <= EPSILON)
  {
    // First segment degenerates into a point
    s = 0;
    t = f / e;  // s = 0 => t = (b*s + f) / e = f / e
    t = G3D::clamp(t, static_cast<Real>(0), static_cast<Real>(1));
  }
  else
  {
    Real c = d1.dot(r);
    if (e <= EPSILON)
    {
        // Second segment degenerates into a point
        t = 0;
        s = G3D::clamp(-c / a, static_cast<Real>(0), static_cast<Real>(1));  // t = 0 => s = (b*t - c) / a = -c / a
    }
    else
    {
      // The general nondegenerate case starts here
      Real b = d1.dot(d2);
      Real denom = a * e - b * b; // Always nonnegative

      // If segments not parallel, compute closest point on L1 to L2, and clamp to segment S1. Else pick arbitrary s (here 0)
      if (denom != 0)
        s = G3D::clamp((b * f - c * e) / denom, static_cast<Real>(0), static_cast<Real>(1));
      else
        s = 0;

      // Compute point on L2 closest to S1(s) using t = Dot((P1+D1*s)-P2,D2) / Dot(D2,D2) = (b*s + f) / e
      t = (b * s + f) / e;

      // If t in [0,1] done. Else clamp t, recompute s for the new value of t using
      // s = Dot((P2+D2*t)-P1,D1) / Dot(D1,D1)= (t*b - c) / a and clamp s to [0, 1]
      if (t < 0)
      {
        t = 0;
        s = G3D::clamp(-c / a, static_cast<Real>(0), static_cast<Real>(1));
      }
      else if (t > 1)
      {
        t = 1;
        s = G3D::clamp((b - c) / a, static_cast<Real>(0), static_cast<Real>(1));
      }
    }
  }

  c1 = p1 + s * d1;
  c2 = p2 + t * d2;
  return (c1 - c2).dot(c1 - c2);
}

using namespace std;

#define FABS(x) ((Real)fabs(x))        /* implement as is fastest on your machine */

/* if USE_EPSILON_TEST is true then we do a check:
         if |dv|<EPSILON then dv=0;
   else no check is done (which is less robust)
*/
#define USE_EPSILON_TEST TRUE
#define EPSILON 0.000001f


/* some macros */
#define CROSS(dest,v1,v2)                      \
              dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
              dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
              dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) dest[0]=v1[0]-v2[0]; dest[1]=v1[1]-v2[1]; dest[2]=v1[2]-v2[2];

#define ADD(dest,v1,v2) dest[0]=v1[0]+v2[0]; dest[1]=v1[1]+v2[1]; dest[2]=v1[2]+v2[2];

#define MULT(dest,v,factor) dest[0]=factor*v[0]; dest[1]=factor*v[1]; dest[2]=factor*v[2];

#define SET(dest,src) dest[0]=src[0]; dest[1]=src[1]; dest[2]=src[2];

/* sort so that a<=b */
#define SORT(a,b)       \
             if(a>b)    \
             {          \
               Real c;  \
               c=a;     \
               a=b;     \
               b=c;     \
             }

#define ISECT(VV0,VV1,VV2,D0,D1,D2,isect0,isect1) \
              isect0=VV0+(VV1-VV0)*D0/(D0-D1);    \
              isect1=VV0+(VV2-VV0)*D0/(D0-D2);


#define COMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,isect0,isect1) \
  if(D0D1>0)                                            \
  {                                                     \
    /* here we know that D0D2<=0 */                     \
    /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
    ISECT(VV2,VV0,VV1,D2,D0,D1,isect0,isect1);          \
  }                                                     \
  else if(D0D2>0)                                       \
  {                                                     \
    /* here we know that d0d1<=0 */                     \
    ISECT(VV1,VV0,VV2,D1,D0,D2,isect0,isect1);          \
  }                                                     \
  else if(D1*D2>0 || D0!=0)                             \
  {                                                     \
    /* here we know that d0d1<=0 or that D0!=0 */       \
    ISECT(VV0,VV1,VV2,D0,D1,D2,isect0,isect1);          \
  }                                                     \
  else if(D1!=0)                                        \
  {                                                     \
    ISECT(VV1,VV0,VV2,D1,D0,D2,isect0,isect1);          \
  }                                                     \
  else if(D2!=0)                                        \
  {                                                     \
    ISECT(VV2,VV0,VV1,D2,D0,D1,isect0,isect1);          \
  }                                                     \
  else                                                  \
  {                                                     \
    /* triangles are coplanar */                        \
    return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2);      \
  }



/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  Real Ax,Ay,Bx,By,Cx,Cy,e,d,f;                \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  Real a,b,c,d0,d1,d2;                      \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0)                               \
  {                                         \
    if(d0*d2>0) return 1;                   \
  }                                         \
}

int coplanar_tri_tri(Real const N[3],Real const V0[3],Real const V1[3],Real const V2[3],
                     Real const U0[3],Real const U1[3],Real const U2[3])
{
   Real A[3];
   short i0,i1;
   /* first project onto an axis-aligned plane, that maximizes the area */
   /* of the triangles, compute indices: i0,i1. */
   A[0]=fabs(N[0]);
   A[1]=fabs(N[1]);
   A[2]=fabs(N[2]);
   if(A[0]>A[1])
   {
      if(A[0]>A[2])
      {
          i0=1;      /* A[0] is greatest */
          i1=2;
      }
      else
      {
          i0=0;      /* A[2] is greatest */
          i1=1;
      }
   }
   else   /* A[0]<=A[1] */
   {
      if(A[2]>A[1])
      {
          i0=0;      /* A[2] is greatest */
          i1=1;
      }
      else
      {
          i0=0;      /* A[1] is greatest */
          i1=2;
      }
    }

    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V1,V2,U0,U1,U2);
    EDGE_AGAINST_TRI_EDGES(V2,V0,U0,U1,U2);

    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0,U0,U1,U2);
    POINT_IN_TRI(U0,V0,V1,V2);

    return 0;
}


int tri_tri_intersect(Real const V0[3],Real const V1[3],Real const V2[3],
                      Real const U0[3],Real const U1[3],Real const U2[3])
{
  Real E1[3],E2[3];
  Real N1[3],N2[3],d1,d2;
  Real du0,du1,du2,dv0,dv1,dv2;
  Real D[3];
  Real isect1[2], isect2[2];
  Real du0du1,du0du2,dv0dv1,dv0dv2;
  short index;
  Real vp0,vp1,vp2;
  Real up0,up1,up2;
  Real b,c,max;

  /* compute plane equation of triangle(V0,V1,V2) */
  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N1,E1,E2);
  d1=-DOT(N1,V0);
  /* plane equation 1: N1.X+d1=0 */

  /* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
  du0=DOT(N1,U0)+d1;
  du1=DOT(N1,U1)+d1;
  du2=DOT(N1,U2)+d1;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
  if(fabs(du0)<EPSILON) du0=0;
  if(fabs(du1)<EPSILON) du1=0;
  if(fabs(du2)<EPSILON) du2=0;
#endif
  du0du1=du0*du1;
  du0du2=du0*du2;

  if(du0du1>0 && du0du2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute plane of triangle (U0,U1,U2) */
  SUB(E1,U1,U0);
  SUB(E2,U2,U0);
  CROSS(N2,E1,E2);
  d2=-DOT(N2,U0);
  /* plane equation 2: N2.X+d2=0 */

  /* put V0,V1,V2 into plane equation 2 */
  dv0=DOT(N2,V0)+d2;
  dv1=DOT(N2,V1)+d2;
  dv2=DOT(N2,V2)+d2;

#if USE_EPSILON_TEST==TRUE
  if(fabs(dv0)<EPSILON) dv0=0;
  if(fabs(dv1)<EPSILON) dv1=0;
  if(fabs(dv2)<EPSILON) dv2=0;
#endif

  dv0dv1=dv0*dv1;
  dv0dv2=dv0*dv2;

  if(dv0dv1>0 && dv0dv2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute direction of intersection line */
  CROSS(D,N1,N2);

  /* compute and index to the largest component of D */
  max=fabs(D[0]);
  index=0;
  b=fabs(D[1]);
  c=fabs(D[2]);
  if(b>max) max=b,index=1;
  if(c>max) max=c,index=2;

  /* this is the simplified projection onto L*/
  vp0=V0[index];
  vp1=V1[index];
  vp2=V2[index];

  up0=U0[index];
  up1=U1[index];
  up2=U2[index];

  /* compute interval for triangle 1 */
  COMPUTE_INTERVALS(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,isect1[0],isect1[1]);

  /* compute interval for triangle 2 */
  COMPUTE_INTERVALS(up0,up1,up2,du0,du1,du2,du0du1,du0du2,isect2[0],isect2[1]);

  SORT(isect1[0],isect1[1]);
  SORT(isect2[0],isect2[1]);

  if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
  return 1;
}


#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1) \
{ \
        if(D0D1>0) \
        { \
                /* here we know that D0D2<=0 */ \
            /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else if(D0D2>0)\
        { \
                /* here we know that d0d1<=0.0f */ \
            A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D1*D2>0 || D0!=0) \
        { \
                /* here we know that d0d1<=0.0f or that D0!=0.0f */ \
                A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
        } \
        else if(D1!=0) \
        { \
                A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D2!=0) \
        { \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else \
        { \
                /* triangles are coplanar */ \
                return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2); \
        } \
}



int NoDivTriTriIsect(Real const V0[3],Real const V1[3],Real const V2[3],
                     Real const U0[3],Real const U1[3],Real const U2[3])
{
  Real E1[3],E2[3];
  Real N1[3],N2[3],d1,d2;
  Real du0,du1,du2,dv0,dv1,dv2;
  Real D[3];
  Real isect1[2], isect2[2];
  Real du0du1,du0du2,dv0dv1,dv0dv2;
  short index;
  Real vp0,vp1,vp2;
  Real up0,up1,up2;
  Real bb,cc,max;
  Real a,b,c,x0,x1;
  Real d,e,f,y0,y1;
  Real xx,yy,xxyy,tmp;

  /* compute plane equation of triangle(V0,V1,V2) */
  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N1,E1,E2);
  d1=-DOT(N1,V0);
  /* plane equation 1: N1.X+d1=0 */

  /* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
  du0=DOT(N1,U0)+d1;
  du1=DOT(N1,U1)+d1;
  du2=DOT(N1,U2)+d1;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
  if(FABS(du0)<EPSILON) du0=0;
  if(FABS(du1)<EPSILON) du1=0;
  if(FABS(du2)<EPSILON) du2=0;
#endif
  du0du1=du0*du1;
  du0du2=du0*du2;

  if(du0du1>0 && du0du2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute plane of triangle (U0,U1,U2) */
  SUB(E1,U1,U0);
  SUB(E2,U2,U0);
  CROSS(N2,E1,E2);
  d2=-DOT(N2,U0);
  /* plane equation 2: N2.X+d2=0 */

  /* put V0,V1,V2 into plane equation 2 */
  dv0=DOT(N2,V0)+d2;
  dv1=DOT(N2,V1)+d2;
  dv2=DOT(N2,V2)+d2;

#if USE_EPSILON_TEST==TRUE
  if(FABS(dv0)<EPSILON) dv0=0;
  if(FABS(dv1)<EPSILON) dv1=0;
  if(FABS(dv2)<EPSILON) dv2=0;
#endif

  dv0dv1=dv0*dv1;
  dv0dv2=dv0*dv2;

  if(dv0dv1>0 && dv0dv2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute direction of intersection line */
  CROSS(D,N1,N2);

  /* compute and index to the largest component of D */
  max=(Real)FABS(D[0]);
  index=0;
  bb=(Real)FABS(D[1]);
  cc=(Real)FABS(D[2]);
  if(bb>max) max=bb,index=1;
  if(cc>max) max=cc,index=2;

  /* this is the simplified projection onto L*/
  vp0=V0[index];
  vp1=V1[index];
  vp2=V2[index];

  up0=U0[index];
  up1=U1[index];
  up2=U2[index];

  /* compute interval for triangle 1 */
  NEWCOMPUTE_INTERVALS(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,a,b,c,x0,x1);

  /* compute interval for triangle 2 */
  NEWCOMPUTE_INTERVALS(up0,up1,up2,du0,du1,du2,du0du1,du0du2,d,e,f,y0,y1);

  xx=x0*x1;
  yy=y0*y1;
  xxyy=xx*yy;

  tmp=a*xxyy;
  isect1[0]=tmp+b*x1*yy;
  isect1[1]=tmp+c*x0*yy;

  tmp=d*xxyy;
  isect2[0]=tmp+e*xx*y1;
  isect2[1]=tmp+f*xx*y0;

  SORT(isect1[0],isect1[1]);
  SORT(isect2[0],isect2[1]);

  if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;
  return 1;
}

/* sort so that a<=b */
#define SORT2(a,b,smallest)       \
             if(a>b)       \
             {             \
               Real c;     \
               c=a;        \
               a=b;        \
               b=c;        \
               smallest=1; \
             }             \
             else smallest=0;


inline void isect2(Real const VTX0[3],Real const VTX1[3],Real const VTX2[3],Real VV0,Real VV1,Real VV2,
                   Real D0,Real D1,Real D2,Real *isect0,Real *isect1,Real isectpoint0[3],Real isectpoint1[3])
{
  Real tmp=D0/(D0-D1);
  Real diff[3];
  *isect0=VV0+(VV1-VV0)*tmp;
  SUB(diff,VTX1,VTX0);
  MULT(diff,diff,tmp);
  ADD(isectpoint0,diff,VTX0);
  tmp=D0/(D0-D2);
  *isect1=VV0+(VV2-VV0)*tmp;
  SUB(diff,VTX2,VTX0);
  MULT(diff,diff,tmp);
  ADD(isectpoint1,VTX0,diff);
}


#if 0
#define ISECT2(VTX0,VTX1,VTX2,VV0,VV1,VV2,D0,D1,D2,isect0,isect1,isectpoint0,isectpoint1) \
              tmp=D0/(D0-D1);                    \
              isect0=VV0+(VV1-VV0)*tmp;          \
              SUB(diff,VTX1,VTX0);               \
              MULT(diff,diff,tmp);               \
              ADD(isectpoint0,diff,VTX0);        \
              tmp=D0/(D0-D2);
/*              isect1=VV0+(VV2-VV0)*tmp;          \ */
/*              SUB(diff,VTX2,VTX0);               \ */
/*              MULT(diff,diff,tmp);               \ */
/*              ADD(isectpoint1,VTX0,diff);          */
#endif

inline int compute_intervals_isectline(Real const VERT0[3],Real const VERT1[3],Real const VERT2[3],
                                       Real VV0,Real VV1,Real VV2,Real D0,Real D1,Real D2,
                                       Real D0D1,Real D0D2,Real *isect0,Real *isect1,
                                       Real isectpoint0[3],Real isectpoint1[3])
{
  if(D0D1>0)
  {
    /* here we know that D0D2<=0 */
    /* that is D0, D1 are on the same side, D2 on the other or on the plane */
    isect2(VERT2,VERT0,VERT1,VV2,VV0,VV1,D2,D0,D1,isect0,isect1,isectpoint0,isectpoint1);
  }
  else if(D0D2>0)
    {
    /* here we know that d0d1<=0 */
    isect2(VERT1,VERT0,VERT2,VV1,VV0,VV2,D1,D0,D2,isect0,isect1,isectpoint0,isectpoint1);
  }
  else if(D1*D2>0 || D0!=0)
  {
    /* here we know that d0d1<=0 or that D0!=0 */
    isect2(VERT0,VERT1,VERT2,VV0,VV1,VV2,D0,D1,D2,isect0,isect1,isectpoint0,isectpoint1);
  }
  else if(D1!=0)
  {
    isect2(VERT1,VERT0,VERT2,VV1,VV0,VV2,D1,D0,D2,isect0,isect1,isectpoint0,isectpoint1);
  }
  else if(D2!=0)
  {
    isect2(VERT2,VERT0,VERT1,VV2,VV0,VV1,D2,D0,D1,isect0,isect1,isectpoint0,isectpoint1);
  }
  else
  {
    /* triangles are coplanar */
    return 1;
  }
  return 0;
}

#define COMPUTE_INTERVALS_ISECTLINE(VERT0,VERT1,VERT2,VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,isect0,isect1,isectpoint0,isectpoint1) \
  if(D0D1>0)                                            \
  {                                                     \
    /* here we know that D0D2<=0 */                     \
    /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
    isect2(VERT2,VERT0,VERT1,VV2,VV0,VV1,D2,D0,D1,&isect0,&isect1,isectpoint0,isectpoint1);          \
  }
#if 0
  else if(D0D2>0)                                       \
  {                                                     \
    /* here we know that d0d1<=0 */                     \
    isect2(VERT1,VERT0,VERT2,VV1,VV0,VV2,D1,D0,D2,&isect0,&isect1,isectpoint0,isectpoint1);          \
  }                                                     \
  else if(D1*D2>0 || D0!=0)                             \
  {                                                     \
    /* here we know that d0d1<=0 or that D0!=0 */       \
    isect2(VERT0,VERT1,VERT2,VV0,VV1,VV2,D0,D1,D2,&isect0,&isect1,isectpoint0,isectpoint1);          \
  }                                                     \
  else if(D1!=0)                                        \
  {                                                     \
    isect2(VERT1,VERT0,VERT2,VV1,VV0,VV2,D1,D0,D2,&isect0,&isect1,isectpoint0,isectpoint1);          \
  }                                                     \
  else if(D2!=0)                                        \
  {                                                     \
    isect2(VERT2,VERT0,VERT1,VV2,VV0,VV1,D2,D0,D1,&isect0,&isect1,isectpoint0,isectpoint1);          \
  }                                                     \
  else                                                  \
  {                                                     \
    /* triangles are coplanar */                        \
    coplanar=1;                                         \
    return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2);      \
  }
#endif

int tri_tri_intersect_with_isectline(Real const V0[3],Real const V1[3],Real const V2[3],
                                     Real const U0[3],Real const U1[3],Real const U2[3],int *coplanar,
                                     Real isectpt1[3],Real isectpt2[3])
{
  Real E1[3],E2[3];
  Real N1[3],N2[3],d1,d2;
  Real du0,du1,du2,dv0,dv1,dv2;
  Real D[3];
  Real isect1[2]={0}, isect2[2]={0};
  Real isectpointA1[3]={0},isectpointA2[3]={0};
  Real isectpointB1[3]={0},isectpointB2[3]={0};
  Real du0du1,du0du2,dv0dv1,dv0dv2;
  short index;
  Real vp0,vp1,vp2;
  Real up0,up1,up2;
  Real b,c,max;
  // Real tmp,diff[3];
  int smallest1,smallest2;

  /* compute plane equation of triangle(V0,V1,V2) */
  SUB(E1,V1,V0);
  SUB(E2,V2,V0);
  CROSS(N1,E1,E2);
  d1=-DOT(N1,V0);
  /* plane equation 1: N1.X+d1=0 */

  /* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
  du0=DOT(N1,U0)+d1;
  du1=DOT(N1,U1)+d1;
  du2=DOT(N1,U2)+d1;

  /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
  if(fabs(du0)<EPSILON) du0=0;
  if(fabs(du1)<EPSILON) du1=0;
  if(fabs(du2)<EPSILON) du2=0;
#endif
  du0du1=du0*du1;
  du0du2=du0*du2;

  if(du0du1>0 && du0du2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute plane of triangle (U0,U1,U2) */
  SUB(E1,U1,U0);
  SUB(E2,U2,U0);
  CROSS(N2,E1,E2);
  d2=-DOT(N2,U0);
  /* plane equation 2: N2.X+d2=0 */

  /* put V0,V1,V2 into plane equation 2 */
  dv0=DOT(N2,V0)+d2;
  dv1=DOT(N2,V1)+d2;
  dv2=DOT(N2,V2)+d2;

#if USE_EPSILON_TEST==TRUE
  if(fabs(dv0)<EPSILON) dv0=0;
  if(fabs(dv1)<EPSILON) dv1=0;
  if(fabs(dv2)<EPSILON) dv2=0;
#endif

  dv0dv1=dv0*dv1;
  dv0dv2=dv0*dv2;

  if(dv0dv1>0 && dv0dv2>0) /* same sign on all of them + not equal 0 ? */
    return 0;              /* no intersection occurs */

  /* compute direction of intersection line */
  CROSS(D,N1,N2);

  /* compute and index to the largest component of D */
  max=fabs(D[0]);
  index=0;
  b=fabs(D[1]);
  c=fabs(D[2]);
  if(b>max) max=b,index=1;
  if(c>max) max=c,index=2;

  /* this is the simplified projection onto L*/
  vp0=V0[index];
  vp1=V1[index];
  vp2=V2[index];

  up0=U0[index];
  up1=U1[index];
  up2=U2[index];

  /* compute interval for triangle 1 */
  *coplanar=compute_intervals_isectline(V0,V1,V2,vp0,vp1,vp2,dv0,dv1,dv2,
                                       dv0dv1,dv0dv2,&isect1[0],&isect1[1],isectpointA1,isectpointA2);
  if(*coplanar) return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2);


  /* compute interval for triangle 2 */
  compute_intervals_isectline(U0,U1,U2,up0,up1,up2,du0,du1,du2,
                              du0du1,du0du2,&isect2[0],&isect2[1],isectpointB1,isectpointB2);

  SORT2(isect1[0],isect1[1],smallest1);
  SORT2(isect2[0],isect2[1],smallest2);

  if(isect1[1]<isect2[0] || isect2[1]<isect1[0]) return 0;

  /* at this point, we know that the triangles intersect */

  if(isect2[0]<isect1[0])
  {
    if(smallest1==0) { SET(isectpt1,isectpointA1); }
    else { SET(isectpt1,isectpointA2); }

    if(isect2[1]<isect1[1])
    {
      if(smallest2==0) { SET(isectpt2,isectpointB2); }
      else { SET(isectpt2,isectpointB1); }
    }
    else
    {
      if(smallest1==0) { SET(isectpt2,isectpointA2); }
      else { SET(isectpt2,isectpointA1); }
    }
  }
  else
  {
    if(smallest2==0) { SET(isectpt1,isectpointB1); }
    else { SET(isectpt1,isectpointB2); }

    if(isect2[1]>isect1[1])
    {
      if(smallest1==0) { SET(isectpt2,isectpointA2); }
      else { SET(isectpt2,isectpointA1); }
    }
    else
    {
      if(smallest2==0) { SET(isectpt2,isectpointB2); }
      else { SET(isectpt2,isectpointB1); }
    }
  }
  return 1;
}

Real
rayTriangleIntersectionTime(Ray3 const & ray, Vector3 const & v0, Vector3 const & edge01, Vector3 const & edge02)
{
  // The code is taken from Dave Eberly's Wild Magic library, v5.3, released under the Boost license:
  // http://www.boost.org/LICENSE_1_0.txt .

  static Real const EPS = 1e-30f;

  Vector3 diff = ray.getOrigin() - v0;
  Vector3 normal = edge01.cross(edge02);

  // Solve Q + t*D = b1*E1 + b2*E2 (Q = diff, D = ray direction, E1 = edge01, E2 = edge02, N = Cross(E1,E2)) by
  //   |Dot(D,N)|*b1 = sign(Dot(D,N))*Dot(D,Cross(Q,E2))
  //   |Dot(D,N)|*b2 = sign(Dot(D,N))*Dot(D,Cross(E1,Q))
  //   |Dot(D,N)|*t = -sign(Dot(D,N))*Dot(Q,N)

  Real DdN = ray.getDirection().dot(normal);
  int sign;
  if (DdN > EPS)
    sign = 1;
  else if (DdN < -EPS)
  {
    sign = -1;
    DdN = -DdN;
  }
  else
  {
    // Ray and triangle are parallel, call it a "no intersection" even if the ray does intersect
    return -1;
  }

  Real DdQxE2 = sign * ray.getDirection().dot(diff.cross(edge02));
  if (DdQxE2 >= 0)
  {
    Real DdE1xQ = sign * ray.getDirection().dot(edge01.cross(diff));
    if (DdE1xQ >= 0)
    {
      if (DdQxE2 + DdE1xQ <= DdN)
      {
        // Line intersects triangle, check if ray does
        Real QdN = -sign * diff.dot(normal);
        if (QdN >= 0)
        {
          // Ray intersects triangle.
          return QdN / DdN;
        }
        // else: t < 0, no intersection
      }
      // else: b1 + b2 > 1, no intersection
    }
    // else: b2 < 0, no intersection
  }
  // else: b1 < 0, no intersection

  return -1;
}

bool isPointInsideTriangle(
    const Vector3&			v0,
    const Vector3&			v1,
    const Vector3&			v2,
    const Vector3&			normal,
    const Vector3&			point,
    float                   b[3],
    Vector3::Axis           primaryAxis) {

    if (primaryAxis == Vector3::DETECT_AXIS) {
        primaryAxis = normal.primaryAxis();
    }

    // Check that the point is within the triangle using a Barycentric
    // coordinate test on a two dimensional plane.
    int i, j;

    switch (primaryAxis) {
    case Vector3::X_AXIS:
        i = Vector3::Y_AXIS;
        j = Vector3::Z_AXIS;
        break;

    case Vector3::Y_AXIS:
        i = Vector3::Z_AXIS;
        j = Vector3::X_AXIS;
        break;

    case Vector3::Z_AXIS:
        i = Vector3::X_AXIS;
        j = Vector3::Y_AXIS;
        break;

    default:
        // This case is here to supress a warning on Linux
        i = j = 0;
        THEA_ASSERT(false, "Should not get here.");
        break;
    }

    // See if all barycentric coordinates are non-negative

    // 2D area via cross product
#   define AREA2(d, e, f)  (((e)[i] - (d)[i]) * ((f)[j] - (d)[j]) - ((f)[i] - (d)[i]) * ((e)[j] - (d)[j]))

    // Area of the polygon
    float area = AREA2(v0, v1, v2);
    if (area == 0) {
        // This triangle has zero area, so the point must not
        // be in it unless the triangle point is the test point.
        return (v0 == point);
    }

    THEA_ASSERT(area != 0, "Area is zero");

    float invArea = 1.0f / area;

    // (avoid normalization until absolutely necessary)
    b[0] = AREA2(point, v1, v2) * invArea;

    if ((b[0] < 0.0f) || (b[0] > 1.0f)) {
        return false;
    }

    b[1] = AREA2(v0,  point, v2) * invArea;
    if ((b[1] < 0.0f) || (b[1] > 1.0f)) {
        return false;
    }

    b[2] = 1.0f - b[0] - b[1];

#   undef AREA2

    return (b[2] >= 0.0f) && (b[2] <= 1.0f);
}

Vector3 closestPointOnLineSegment(
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& point) {

    const Vector3& edge       = (v1 - v0);
    float          edgeLength = edge.magnitude();

    if (edgeLength == 0) {
        // The line segment is a point
        return v0;
    }

    return closestPointOnLineSegment(v0, v1, edge / edgeLength, edgeLength, point);
}


Vector3 closestPointOnLineSegment(
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& edgeDirection,
    const float    edgeLength,
    const Vector3& point) {

    THEA_ASSERT((v1 - v0).direction().fuzzyEq(edgeDirection), "Edge direction not consistent with vertices");
    THEA_ASSERT(G3D::fuzzyEq((v1 - v0).magnitude(), edgeLength), "Edge length not consistent with vertices");

    // Vector towards the point
    const Vector3& c = point - v0;

    // Projected onto the edge itself
    float t = edgeDirection.dot(c);

    if (t <= 0) {
        // Before the start
        return v0;
    } else if (t >= edgeLength) {
        // After the end
        return v1;
    } else {
        // At distance t along the edge
        return v0 + edgeDirection * t;
    }
}

Vector3 closestPointOnTrianglePerimeter(
    const Vector3&			v0,
    const Vector3&			v1,
    const Vector3&			v2,
    const Vector3&			point) {

    Vector3 v[3] = {v0, v1, v2};
    Vector3 edgeDirection[3] = {(v1 - v0), (v2 - v1), (v0 - v2)};
    float   edgeLength[3];

    for (int i = 0; i < 3; ++i) {
        edgeLength[i] = edgeDirection[i].magnitude();
        edgeDirection[i] /= edgeLength[i];
    }

    int edgeIndex;
    return closestPointOnTrianglePerimeter(v, edgeDirection, edgeLength, point, edgeIndex);
}


Vector3 closestPointOnTrianglePerimeter(
    const Vector3   v[3],
    const Vector3   edgeDirection[3],
    const float     edgeLength[3],
    const Vector3&  point,
    int&            edgeIndex) {

    // Closest point on segment from v[i] to v[i + 1]
    Vector3 r[3];

    // Distance squared from r[i] to point
    float d[3];

    // Index of the next point
    static const int next[] = {1, 2, 0};

    for (int i = 0; i < 3; ++i) {
        r[i] = closestPointOnLineSegment(v[i], v[next[i]], edgeDirection[i], edgeLength[i], point);
        d[i] = (r[i] - point).squaredMagnitude();
    }

    if (d[0] < d[1]) {
        if (d[0] < d[2]) {
            // Between v0 and v1
            edgeIndex = 0;
        } else {
            // Between v2 and v0
            edgeIndex = 2;
        }
    } else {
        if (d[1] < d[2]) {
            // Between v1 and v2
            edgeIndex = 1;
        } else {
            // Between v2 and v0
            edgeIndex = 2;
        }
    }

#   ifdef G3D_DEBUG
    {
        Vector3 diff = r[edgeIndex] - v[edgeIndex];
        THEA_ASSERT(fuzzyEq(diff.direction().dot(edgeDirection[edgeIndex]), 1.0f) ||
            diff.fuzzyEq(Vector3::zero()), "Point not on correct triangle edge");
        float frac = diff.dot(edgeDirection[edgeIndex])/edgeLength[edgeIndex];
        THEA_ASSERT(frac >= -0.000001, "Point off low side of edge.");
        THEA_ASSERT(frac <= 1.000001, "Point off high side of edge.");
    }
#   endif

    return r[edgeIndex];
}

} // namespace Triangle3Internal

} // namespace Thea
