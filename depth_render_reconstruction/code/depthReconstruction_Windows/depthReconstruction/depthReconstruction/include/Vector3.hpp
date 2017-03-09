/**
  @file Vector3.h

  3D vector class

  @maintainer Morgan McGuire, http://graphics.cs.williams.edu

  @created 2001-06-02
  @edited  2009-11-01
  Copyright 2000-2009, Morgan McGuire.
  All rights reserved.
 */

#ifndef G3D_Vector3_hpp
#define G3D_Vector3_hpp

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>

#if (defined(_DEBUG) || !defined(NDEBUG))
#  define THEA_ASSERT(cond, msg) \
     if (!(cond)) \
     { throw std::runtime_error(msg); }
#else
#  define THEA_ASSERT(cond, msg) {}
#endif

namespace G3D {

/**
 Computes an appropriate epsilon for comparing a and b.
 */
inline double eps(double a, double b) {
    static double const fuzzyEpsilon = 0.00001;

    // For a and b to be nearly equal, they must have nearly
    // the same magnitude.  This means that we can ignore b
    // since it either has the same magnitude or the comparison
    // will fail anyway.
    (void)b;
    const double aa = std::fabs(a) + 1.0;
    if (aa > 1.0e+300) {
        return fuzzyEpsilon;
    } else {
        return fuzzyEpsilon * aa;
    }
}

inline bool fuzzyEq(double a, double b) {
    return (a == b) || (std::fabs(a - b) <= eps(a, b));
}

inline bool fuzzyNe(double a, double b) {
    return ! fuzzyEq(a, b);
}

template <typename T> T clamp(T x, T lo, T hi)
{
  if (x < lo)
    return x;
  else if (x > hi)
    return hi;
  else
    return x;
}

template <typename T> T square(T t) { return t * t; }

/**
  <B>Warning</B>

 Do not subclass-- this implementation makes assumptions about the
 memory layout.
 */
class Vector3 {
public:

    // coordinates
    float x, y, z;

private:

    // Hidden operators
    bool operator<(const Vector3&) const;
    bool operator>(const Vector3&) const;
    bool operator<=(const Vector3&) const;
    bool operator>=(const Vector3&) const;

public:
    /** Initializes to zero */
    Vector3();

    Vector3(float _x, float _y, float _z);
    explicit Vector3(float coordinate[3]);
    explicit Vector3(double coordinate[3]);

    // access vector V as V[0] = V.x, V[1] = V.y, V[2] = V.z
    //
    // WARNING.  These member functions rely on
    // (1) Vector3 not having virtual functions
    // (2) the data packed in a 3*sizeof(float) memory block
    const float& operator[] (int i) const;
    float& operator[] (int i);

    enum Axis {X_AXIS=0, Y_AXIS=1, Z_AXIS=2, DETECT_AXIS=-1};

    /**
     Returns the largest dimension.  Particularly convenient for determining
     which plane to project a triangle onto for point-in-polygon tests.
     */
    Axis primaryAxis() const
    {
        Axis a = X_AXIS;

        double nx = std::fabs(x);
        double ny = std::fabs(y);
        double nz = std::fabs(z);

        if (nx > ny) {
            if (nx > nz) {
                a = X_AXIS;
            } else {
                a = Z_AXIS;
            }
        } else {
            if (ny > nz) {
                a = Y_AXIS;
            } else {
                a = Z_AXIS;
            }
        }

        return a;
    }

    // assignment and comparison
    Vector3& operator= (const Vector3& rkVector);
    bool operator== (const Vector3& rkVector) const;
    bool operator!= (const Vector3& rkVector) const;
    size_t hashCode() const;
    bool fuzzyEq(const Vector3& other) const;
    bool fuzzyNe(const Vector3& other) const;

    /** Returns true if this vector has length ~= 0 */
    bool isZero() const;

    /** Returns true if this vector has length ~= 1 */
    bool isUnit() const;

    // arithmetic operations
    Vector3 operator+ (const Vector3& v) const;
    Vector3 operator- (const Vector3& v) const;
    Vector3 operator* (float s) const;
    inline Vector3 operator/ (float s) const {
        return *this * (1.0f / s);
    }
    Vector3 operator* (const Vector3& v) const;
    Vector3 operator/ (const Vector3& v) const;
    Vector3 operator- () const;

    // arithmetic updates
    Vector3& operator+= (const Vector3& v);
    Vector3& operator-= (const Vector3& v);
    Vector3& operator*= (float s);
    inline Vector3& operator/= (float s) {
        return (*this *= (1.0f / s));
    }
    Vector3& operator*= (const Vector3& v);
    Vector3& operator/= (const Vector3& v);

    /** Same as magnitude */
    float length() const;

    float magnitude() const;

    /**
     The result is a nan vector if the length is almost zero.
     */
    Vector3 direction() const;

    /**
     Reflect this vector about the (not necessarily unit) normal.
     Assumes that both the before and after vectors point away from
     the base of the normal.

     Note that if used for a collision or ray reflection you
     must negate the resulting vector to get a direction pointing
     <I>away</I> from the collision.

     <PRE>
       V'    N      V

         r   ^   -,
          \  |  /
            \|/
     </PRE>

     See also Vector3::reflectionDirection
     */
    Vector3 reflectAbout(const Vector3& normal) const
    {
      Vector3 N = normal.direction();

      // 2 * normal.dot(this) * normal - this
      return N * 2 * this->dot(N) - *this;
    }

    /**
      See also G3D::Ray::reflect.
      The length is 1.
     <PRE>
       V'    N       V

         r   ^    /
          \  |  /
            \|'-
     </PRE>
     */
    Vector3 reflectionDirection(const Vector3& normal) const
    {
      return -reflectAbout(normal).direction();
    }

    /**
     Returns Vector3::zero() if the length is nearly zero, otherwise
     returns a unit vector.
     */
    inline Vector3 directionOrZero() const {
        float mag = magnitude();
        if (mag < 0.0000001f) {
            return Vector3::zero();
        } else if (mag < 1.00001f && mag > 0.99999f) {
            return *this;
        } else {
            return *this * (1.0f / mag);
        }
    }

    /**
     Returns the direction of a refracted ray,
     where iExit is the index of refraction for the
     previous material and iEnter is the index of refraction
     for the new material.  Like Vector3::reflectionDirection,
     the result has length 1 and is
     pointed <I>away</I> from the intersection.

     Returns Vector3::zero() in the case of total internal refraction.

     @param iOutside The index of refraction (eta) outside
     (on the <I>positive</I> normal side) of the surface.

     @param iInside The index of refraction (eta) inside
     (on the <I>negative</I> normal side) of the surface.

     See also G3D::Ray::refract.
     <PRE>
              N      V

              ^    /
              |  /
              |'-
          __--
     V'<--
     </PRE>
     */
    Vector3 refractionDirection(
        const Vector3&  normal,
        float           iInside,
        float           iOutside) const;

    /** Synonym for direction */
    inline Vector3 unit() const {
        return direction();
    }

    /** Same as squaredMagnitude */
    float squaredLength() const;

    float squaredMagnitude () const;

    float dot(const Vector3& rkVector) const;

    float unitize(float tolerance = 1e-30);

    /** Cross product.  Note that two cross products in a row
        can be computed more cheaply: v1 x (v2 x v3) = (v1 dot v3) v2  - (v1 dot v2) v3.
      */
    Vector3 cross(const Vector3& rkVector) const;
    Vector3 unitCross(const Vector3& rkVector) const;

    /**
     Returns a matrix such that v.cross() * w = v.cross(w).
     <PRE>
     [ 0  -v.z  v.y ]
     [ v.z  0  -v.x ]
     [ -v.y v.x  0  ]
     </PRE>
     */
    class Matrix3 cross() const;

    Vector3 min(const Vector3 &v) const;
    Vector3 max(const Vector3 &v) const;

    /** Smallest element */
    inline float min() const {
        return std::min(std::min(x, y), z);
    }

    /** Largest element */
    inline float max() const {
        return std::max(std::max(x, y), z);
    }

    std::string toString() const
    {
      std::ostringstream oss;
      oss << '(' << x << ", " << y << ", " << z << ')';
      return oss.str();
    }

    inline Vector3 clamp(const Vector3& low, const Vector3& high) const {
        return Vector3(
            G3D::clamp(x, low.x, high.x),
            G3D::clamp(y, low.y, high.y),
            G3D::clamp(z, low.z, high.z));
    }

    inline Vector3 clamp(float low, float high) const {
        return Vector3(
            G3D::clamp(x, low, high),
            G3D::clamp(y, low, high),
            G3D::clamp(z, low, high));
    }

    /**
     Linear interpolation
     */
    inline Vector3 lerp(const Vector3& v, float alpha) const {
        return (*this) + (v - *this) * alpha;
    }

    /** Gram-Schmidt orthonormalization. */
    static void orthonormalize (Vector3 akVector[3])
    {
        // If the input vectors are v0, v1, and v2, then the Gram-Schmidt
        // orthonormalization produces vectors u0, u1, and u2 as follows,
        //
        //   u0 = v0/|v0|
        //   u1 = (v1-(u0*v1)u0)/|v1-(u0*v1)u0|
        //   u2 = (v2-(u0*v2)u0-(u1*v2)u1)/|v2-(u0*v2)u0-(u1*v2)u1|
        //
        // where |A| indicates length of vector A and A*B indicates dot
        // product of vectors A and B.

        // compute u0
        akVector[0].unitize();

        // compute u1
        float fDot0 = akVector[0].dot(akVector[1]);
        akVector[1] -= akVector[0] * fDot0;
        akVector[1].unitize();

        // compute u2
        float fDot1 = akVector[1].dot(akVector[2]);
        fDot0 = akVector[0].dot(akVector[2]);
        akVector[2] -= akVector[0] * fDot0 + akVector[1] * fDot1;
        akVector[2].unitize();
    }

    /** Input W must be initialize to a nonzero vector, output is {U,V,W}
        an orthonormal basis.  A hint is provided about whether or not W
        is already unit length.
        @deprecated Use getTangents
    */
    static void generateOrthonormalBasis (Vector3& rkU, Vector3& rkV,
                                          Vector3& rkW, bool bUnitLengthW = true)
    {
        if ( !bUnitLengthW )
            rkW.unitize();

        if ( std::fabs(rkW.x) >= std::fabs(rkW.y)
                && std::fabs(rkW.x) >= std::fabs(rkW.z) ) {
            rkU.x = -rkW.y;
            rkU.y = + rkW.x;
            rkU.z = 0.0;
        } else {
            rkU.x = 0.0;
            rkU.y = + rkW.z;
            rkU.z = -rkW.y;
        }

        rkU.unitize();
        rkV = rkW.cross(rkU);
    }

    inline float sum() const {
        return x + y + z;
    }

    inline float average() const {
        return sum() / 3.0f;
    }

    // Special values.
    static const Vector3& zero()     { static const Vector3 v(0, 0, 0); return v; }
    static const Vector3& one()      { static const Vector3 v(1, 1, 1); return v; }
    static const Vector3& unitX()    { static const Vector3 v(1, 0, 0); return v; }
    static const Vector3& unitY()    { static const Vector3 v(0, 1, 0); return v; }
    static const Vector3& unitZ()    { static const Vector3 v(0, 0, 1); return v; }

    /** Smallest (most negative) representable vector */
    static const Vector3& minFinite()
    {
      static float const MAX_FLOAT = std::numeric_limits<float>::max();
      static const Vector3 v(-MAX_FLOAT, -MAX_FLOAT, -MAX_FLOAT);
      return v;
    }

    /** Largest representable vector */
    static const Vector3& maxFinite()
    {
      static float const MAX_FLOAT = std::numeric_limits<float>::max();
      static const Vector3 v(MAX_FLOAT, MAX_FLOAT, MAX_FLOAT);
      return v;
    }


    /** Creates two orthonormal tangent vectors X and Y such that
        if Z = this, X x Y = Z.*/
    inline void getTangents(Vector3& X, Vector3& Y) const {
        THEA_ASSERT(G3D::fuzzyEq(length(), 1.0f), "makeAxes requires Z to have unit length");

        // Choose another vector not perpendicular
        X = (std::fabs(x) < 0.9f) ? Vector3::unitX() : Vector3::unitY();

        // Remove the part that is parallel to Z
        X -= *this * this->dot(X);
        X /= X.length();

        Y = this->cross(X);
    }

    /** Can be passed to ignore a vector3 parameter */
    static Vector3& ignore();
};

inline G3D::Vector3 operator*(float s, const G3D::Vector3& v) {
    return v * s;
}

inline G3D::Vector3 operator*(double s, const G3D::Vector3& v) {
    return v * (float)s;
}

inline G3D::Vector3 operator*(int s, const G3D::Vector3& v) {
    return v * (float)s;
}

//----------------------------------------------------------------------------
inline Vector3::Vector3() : x(0.0f), y(0.0f), z(0.0f) {
}

//----------------------------------------------------------------------------

inline Vector3::Vector3 (float fX, float fY, float fZ) : x(fX), y(fY), z(fZ) {
}

//----------------------------------------------------------------------------
inline Vector3::Vector3 (float V[3]) : x(V[0]), y(V[1]), z(V[2]){
}

//----------------------------------------------------------------------------
inline Vector3::Vector3 (double V[3]) : x((float)V[0]), y((float)V[1]), z((float)V[2]){
}

//----------------------------------------------------------------------------
inline const float& Vector3::operator[] (int i) const {
    return ((float*)this)[i];
}

inline float& Vector3::operator[] (int i) {
    return ((float*)this)[i];
}


//----------------------------------------------------------------------------
inline Vector3& Vector3::operator= (const Vector3& rkVector) {
    x = rkVector.x;
    y = rkVector.y;
    z = rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------

inline bool Vector3::fuzzyEq(const Vector3& other) const {
    return G3D::fuzzyEq((*this - other).squaredMagnitude(), 0);
}

//----------------------------------------------------------------------------

inline bool Vector3::fuzzyNe(const Vector3& other) const {
    return G3D::fuzzyNe((*this - other).squaredMagnitude(), 0);
}

//----------------------------------------------------------------------------
inline bool Vector3::operator== (const Vector3& rkVector) const {
    return ( x == rkVector.x && y == rkVector.y && z == rkVector.z );
}

//----------------------------------------------------------------------------
inline bool Vector3::operator!= (const Vector3& rkVector) const {
    return ( x != rkVector.x || y != rkVector.y || z != rkVector.z );
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator+ (const Vector3& rkVector) const {
    return Vector3(x + rkVector.x, y + rkVector.y, z + rkVector.z);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator- (const Vector3& rkVector) const {
    return Vector3(x - rkVector.x, y - rkVector.y, z - rkVector.z);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator* (const Vector3& rkVector) const {
    return Vector3(x * rkVector.x, y * rkVector.y, z * rkVector.z);
}

inline Vector3 Vector3::operator*(float f) const {
    return Vector3(x * f, y * f, z * f);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator/ (const Vector3& rkVector) const {
    return Vector3(x / rkVector.x, y / rkVector.y, z / rkVector.z);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::operator- () const {
    return Vector3(-x, -y, -z);
}

//----------------------------------------------------------------------------
inline Vector3& Vector3::operator+= (const Vector3& rkVector) {
    x += rkVector.x;
    y += rkVector.y;
    z += rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------
inline Vector3& Vector3::operator-= (const Vector3& rkVector) {
    x -= rkVector.x;
    y -= rkVector.y;
    z -= rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------
inline Vector3& Vector3::operator*= (float fScalar) {
    x *= fScalar;
    y *= fScalar;
    z *= fScalar;
    return *this;
}

//----------------------------------------------------------------------------
inline Vector3& Vector3::operator*= (const Vector3& rkVector) {
    x *= rkVector.x;
    y *= rkVector.y;
    z *= rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------
inline Vector3& Vector3::operator/= (const Vector3& rkVector) {
    x /= rkVector.x;
    y /= rkVector.y;
    z /= rkVector.z;
    return *this;
}

//----------------------------------------------------------------------------
inline float Vector3::squaredMagnitude () const {
    return x*x + y*y + z*z;
}

//----------------------------------------------------------------------------
inline float Vector3::squaredLength () const {
    return squaredMagnitude();
}

//----------------------------------------------------------------------------
inline float Vector3::magnitude() const {
    return std::sqrt(x*x + y*y + z*z);
}

//----------------------------------------------------------------------------
inline float Vector3::length() const {
    return magnitude();
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::direction () const {
    const float lenSquared = squaredMagnitude();
    const float invSqrt = 1.0f / std::sqrt(lenSquared);
    return Vector3(x * invSqrt, y * invSqrt, z * invSqrt);
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
inline float Vector3::dot (const Vector3& rkVector) const {
    return x*rkVector.x + y*rkVector.y + z*rkVector.z;
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::cross (const Vector3& rkVector) const {
    return Vector3(y*rkVector.z - z*rkVector.y, z*rkVector.x - x*rkVector.z,
                   x*rkVector.y - y*rkVector.x);
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::unitCross (const Vector3& rkVector) const {
    Vector3 kCross(y*rkVector.z - z*rkVector.y, z*rkVector.x - x*rkVector.z,
                   x*rkVector.y - y*rkVector.x);
    kCross.unitize();
    return kCross;
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::min(const Vector3 &v) const {
    return Vector3(std::min(v.x, x), std::min(v.y, y), std::min(v.z, z));
}

//----------------------------------------------------------------------------
inline Vector3 Vector3::max(const Vector3 &v) const {
    return Vector3(std::max(v.x, x), std::max(v.y, y), std::max(v.z, z));
}

//----------------------------------------------------------------------------
inline bool Vector3::isZero() const {
    return G3D::fuzzyEq(squaredMagnitude(), 0.0f);
}

//----------------------------------------------------------------------------

inline bool Vector3::isUnit() const {
    return G3D::fuzzyEq(squaredMagnitude(), 1.0f);
}

inline Vector3 Vector3::refractionDirection(
    const Vector3&  normal,
    float           iInside,
    float           iOutside) const
{

    // From pg. 24 of Henrik Wann Jensen. Realistic Image Synthesis
    // Using Photon Mapping.  AK Peters. ISBN: 1568811470. July 2001.

    // Invert the directions from Wann Jensen's formulation
    // and normalize the vectors.
    const Vector3 W = -direction();
    Vector3 N = normal.direction();

    float h1 = iOutside;
    float h2 = iInside;

    if (normal.dot(*this) > 0.0f) {
        h1 = iInside;
        h2 = iOutside;
        N  = -N;
    }

    const float hRatio = h1 / h2;
    const float WdotN = W.dot(N);

    float det = 1.0f - (float)square(hRatio) * (1.0f - (float)square(WdotN));

    if (det < 0) {
        // Total internal reflection
        return Vector3::zero();
    } else {
        return -hRatio * (W - WdotN * N) - N * std::sqrt(det);
    }
}

} // namespace G3D


#endif
