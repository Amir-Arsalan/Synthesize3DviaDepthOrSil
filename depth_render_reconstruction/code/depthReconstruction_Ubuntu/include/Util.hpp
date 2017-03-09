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

#ifndef __Thea_Util_hpp__
#define __Thea_Util_hpp__

#include "Vector3.hpp"
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>

namespace Thea {

typedef float Real;
using G3D::Vector3;
using std::size_t;

#define THEA_ENUM_CLASS_BODY(name)                                                                                            \
    public:                                                                                                                   \
      name() {}                                                                                                               \
      template <typename T> explicit name(T value_) : value(static_cast<Value>(value_)) {}                                    \
      name(Value value_) : value(value_) {}                                                                                   \
      operator Value() const { return value; }                                                                                \
      template <typename T> operator T() const { return static_cast<T>(value); }                                              \
      bool operator==(Value other) const { return value == other; }                                                           \
      bool operator!=(Value other) const { return value != other; }                                                           \
      bool operator==(name const & other) const { return value == other.value; }                                              \
      bool operator!=(name const & other) const { return value != other.value; }                                              \
                                                                                                                              \
    private:                                                                                                                  \
      Value value;

/** Coordinate axes upto 4D (enum class). */
struct CoordinateAxis
{
  /** Supported values. */
  enum Value
  {
    X = G3D::Vector3::X_AXIS,  ///< The X axis.
    Y = G3D::Vector3::Y_AXIS,  ///< The Y axis.
    Z = G3D::Vector3::Z_AXIS,  ///< The Z axis.
    W                          ///< The W axis.
  };

  THEA_ENUM_CLASS_BODY(CoordinateAxis)
};

// Adapted from the Boost type_traits examples.
namespace FastCopyInternal {

template <typename I1, typename I2, bool b>
I2
fastCopyImpl(I1 first, I1 last, I2 out, boost::integral_constant<bool, b> const &)
{
  while (first != last) *(out++) = *(first++);
  return out;
}

template <typename T>
T *
fastCopyImpl(T const * first, T const * last, T * out, boost::true_type const &)
{
   memcpy(out, first, (last - first) * sizeof(T));
   return out + (last - first);
}

// Same semantics as std::copy, calls memcpy where appropriate.
template <typename I1, typename I2, bool b>
I2
fastCopyBackwardImpl(I1 first, I1 last, I2 out, boost::integral_constant<bool, b> const &)
{
  while (last != first) *(--out) = *(--last);
  return out;
}

template <typename T>
T *
fastCopyBackwardImpl(T const * first, T const * last, T * out, boost::true_type const &)
{
   memmove(out, first, (last - first) * sizeof(T));
   return out;
}

} // namespace FastCopyInternal

/**
 * A version of <tt>std::copy</tt> that calls <tt>memcpy</tt> where appropriate (if the class has a trivial assignment operator)
 * for speed.
 *
 * To take advantage of fast copying, specialize boost::has_trivial_assign to return true for the value type.
 */
template <typename I1, typename I2>
inline I2
fastCopy(I1 first, I1 last, I2 out)
{
  //
  // We can copy with memcpy if T has a trivial assignment operator,
  // and if the iterator arguments are actually pointers (this last
  // requirement we detect with overload resolution):
  //
  typedef typename std::iterator_traits<I1>::value_type value_type;
  return FastCopyInternal::fastCopyImpl(first, last, out, boost::has_trivial_assign<value_type>());
}

/**
 * A version of <tt>std::copy_backward</tt> that calls <tt>memmove</tt> where appropriate (if the class has a trivial assignment
 * operator) for speed.
 *
 * To take advantage of fast copying, specialize boost::has_trivial_assign to return true for the value type.
 */
template <typename I1, typename I2>
inline I2
fastCopyBackward(I1 first, I1 last, I2 out)
{
  typedef typename std::iterator_traits<I1>::value_type value_type;
  return FastCopyInternal::fastCopyBackwardImpl(first, last, out, boost::has_trivial_assign<value_type>());
}

/**
 * A description of the intersection point of a ray with an object. Specifies the hit time and the normal at the intersection
 * point.
 */
class RayIntersection3
{
  private:
    Real time;
    bool has_normal;
    Vector3 normal;

  public:
    /** Constructor. */
    RayIntersection3(Real time_ = -1, Vector3 const * normal_ = NULL)
    : time(time_), has_normal(normal_ != NULL), normal(normal_ ? *normal_ : Vector3::zero())
    {}

    /** Check if the intersection is valid. */
    bool isValid() const { return time >= 0; }

    /** Get the intersection time. */
    Real getTime() const { return time; }

    /** Set the intersection time. */
    void setTime(Real time_) { time = time_; }

    /** Check if the normal at the intersection point is known. */
    bool hasNormal() const { return has_normal; }

    /** Get the normal at the intersection point. The return value is undefined if hasNormal() returns false. */
    Vector3 const & getNormal() const { return normal; }

    /** Set the normal at the intersection point. hasNormal() will subsequently return true. */
    void setNormal(Vector3 const & normal_) { normal = normal_; has_normal = true; }

}; // class RayIntersection3

/** A ray in 3-space, having an originating point and a direction vector (not necessarily unit length). */
class Ray3
{
  public:
    /** Default constructor. Does not initialize anything. */
    Ray3() {}

    /** Initialize with an origin and a direction. */
    Ray3(Vector3 const & origin_, Vector3 direction_) : origin(origin_), direction(direction_) {}

    /** Copy constructor. */
    Ray3(Ray3 const & src) : origin(src.origin), direction(src.direction) {}

    /** Get the origin of the ray. */
    Vector3 const & getOrigin() const { return origin; }

    /** Set the origin of the ray. */
    void setOrigin(Vector3 const & origin_) { origin = origin_; }

    /** Get the direction of the ray. */
    Vector3 const & getDirection() const { return direction; }

    /** Set the direction of the ray. */
    void setDirection(Vector3 const & direction_) { direction = direction_; }

    /** Make the direction vector unit length. */
    void normalizeDirection() { direction.unitize(); }

    /**
     * Get a parametrized point on the ray.
     *
     * @return getOrigin() + \a t * getDirection()
     */
    Vector3 getPoint(Real t) const { return origin + t * direction; }

    /** Get the distance of a point from the ray. */
    Real distance(Vector3 const & p) const { return (closestPoint(p) - p).length(); }

    /** Get the square of the distance of a point from the ray. */
    Real squaredDistance(Vector3 const & p) const { return (closestPoint(p) - p).squaredLength(); }

    /** Get the the point on the ray closest to a given point. */
    Vector3 closestPoint(Vector3 const & p) const
    {
      Real t = (p - origin).dot(direction);  // we'll normalize direction later
      if (t < 0)
        return origin;
      else
      {
        // direction has to be normalized twice -- once during the dot product above and once in the scaling below. We can combine
        // the two divisions and avoid the square roots.
        return origin + t * direction / direction.squaredLength();
      }
    }

    /** Get a textual representation of the ray. */
    std::string toString() const
    {
      std::ostringstream oss;
      oss << "[origin: " << origin.toString() << ", direction: " << direction.toString() << ']';
      return oss.str();
    }

  private:
    Vector3 origin;  ///< Origin of the ray.
    Vector3 direction;  ///< Direction of the ray.

}; // class Ray3

class AxisAlignedBox3
{
  private:
    Vector3 lo, hi;

  public:
    AxisAlignedBox3() {}
    AxisAlignedBox3(Vector3 const & lo_, Vector3 const & hi_) : lo(lo_), hi(hi_) {}

    Vector3 const & low() const { return lo; }
    Vector3 const & high() const { return hi; }

    void set(Vector3 const & lo_, Vector3 const & hi_)
    {
      lo = lo_;
      hi = hi_;
    }

    /** Returns the centroid of the box. */
    Vector3 center() const
    {
      return (lo + hi) * 0.5;
    }

    Vector3 extent() const
    {
      return hi - lo;
    }

    /** Corner with index i in {0, 1, ..., 7}. */
    Vector3 corner(int index) const
    {
      // default constructor inits all components to 0
      Vector3 v;

      switch (index)
      {
        case 0:
            v.x = lo.x;
            v.y = lo.y;
            v.z = hi.z;
            break;

        case 1:
            v.x = hi.x;
            v.y = lo.y;
            v.z = hi.z;
            break;

        case 2:
            v.x = hi.x;
            v.y = hi.y;
            v.z = hi.z;
            break;

        case 3:
            v.x = lo.x;
            v.y = hi.y;
            v.z = hi.z;
            break;

        case 4:
            v.x = lo.x;
            v.y = lo.y;
            v.z = lo.z;
            break;

        case 5:
            v.x = hi.x;
            v.y = lo.y;
            v.z = lo.z;
            break;

        case 6:
            v.x = hi.x;
            v.y = hi.y;
            v.z = lo.z;
            break;

        case 7:
            v.x = lo.x;
            v.y = hi.y;
            v.z = lo.z;
            break;

        default:
            THEA_ASSERT(false, "AxisAlignedBox3: Invalid corner index");
            break;
      }

      return v;
    }

    /* Grows to include the bounds of a. */
    void merge(AxisAlignedBox3 const & a)
    {
      lo = lo.min(a.lo);
      hi = hi.max(a.hi);
    }

    void merge(Vector3 const & a)
    {
      lo = lo.min(a);
      hi = hi.max(a);
    }

    /** less than or equal to containment */
    bool contains(AxisAlignedBox3 const & other) const
    {
      return
          (other.hi.x <= hi.x) &&
          (other.hi.y <= hi.y) &&
          (other.hi.z <= hi.z) &&
          (other.lo.x >= lo.x) &&
          (other.lo.y >= lo.y) &&
          (other.lo.z >= lo.z);
    }

    bool contains(Vector3 const & point) const
    {
      return
          (point.x >= lo.x) &&
          (point.y >= lo.y) &&
          (point.z >= lo.z) &&
          (point.x <= hi.x) &&
          (point.y <= hi.y) &&
          (point.z <= hi.z);
    }

    /** Test if this box intersects (i.e. contains) a point. */
    bool intersects(Vector3 const & p) const { return contains(p); }

    /** Test if this box intersects another box. */
    bool intersects(AxisAlignedBox3 const & other) const
    {
      // Must be overlap along all three axes.
      // Try to find a separating axis.

      for (int a = 0; a < 3; ++a) {

          //     |--------|
          // |------|

          if ((lo[a] > other.hi[a]) ||
              (hi[a] < other.lo[a])) {
              return false;
          }
      }

      return true;
    }

    /** Scale the box by a linear factor relative to its center. */
    void scaleCentered(Real scale)
    {
      Vector3 c = center();
      set(scale * (low() - c) + c, scale * (high() - c) + c);
    }

    /** Get the closest distance to a point. */
    Real distance(Vector3 const & point) const { return std::sqrt(squaredDistance(point)); }

    /** Get the closest distance between this box and another one. */
    Real distance(AxisAlignedBox3 const & other) const { return std::sqrt(squaredDistance(other)); }

    /** Get the square of the closest distance to a point. */
    Real squaredDistance(Vector3 const & point) const
    {
      Vector3 vmax = point.min(high());
      Vector3 vmin = point.max(low());

      return (vmin - vmax).squaredLength();
    }

    /** Get the square of the closest distance between this box and another one. */
    Real squaredDistance(AxisAlignedBox3 const & other) const
    {
      Vector3 vmax = high().min(other.high());
      Vector3 vmin = low().max(other.low());

      // Each coord of vmax is greater than the corresponding coord of vmin iff the ranges intersect

      Vector3 vdiff = vmin - vmax;  // any axes with overlap have negative values here
      vdiff = vdiff.max(Vector3::zero());  // overlap axes have zero separation

      return vdiff.squaredLength();
    }

    /** Get the largest distance to a point. */
    Real maxDistance(Vector3 const & point) const { return std::sqrt(squaredMaxDistance(point)); }

    /** Get the square of the largest distance to a point. */
    Real squaredMaxDistance(Vector3 const & point) const
    {
      Vector3 vdiff = (high() - point).max(point - low());
      return vdiff.squaredLength();
    }

    bool rayIntersects(Ray3 const & ray, Real max_time = -1) const
    {
      // TODO: Speed this up: see G3D::Intersect::rayAABox
      return (rayIntersectionTime(ray) >= 0);
    }

    Real rayIntersectionTime(Ray3 const & ray, Real max_time = -1) const
    {
      // TODO: Speed this up: see G3D::Intersect::rayAABox
      RayIntersection3 isec = rayIntersection(ray, max_time);
      return isec.getTime();
    }

    RayIntersection3 rayIntersection(Ray3 const & ray, Real max_time = -1) const;

    std::string toString() const
    {
      std::ostringstream oss;
      oss << '[' << low().toString() << ", " << high().toString() << ']';
      return oss.str();
    }

    static AxisAlignedBox3 const & zero()
    {
      static AxisAlignedBox3 const z(Vector3::zero(), Vector3::zero());
      return z;
    }

}; // class AxisAlignedBox3

// From G3D::CollisionDetection
inline bool collisionLocationForMovingPointFixedAABox(
    const Vector3&          origin,
    const Vector3&          dir,
    const AxisAlignedBox3&  box,
    Vector3&                location,
    bool&                   Inside,
    Vector3&                normal) {

    // Integer representation of a floating-point value.
    #define IR(x)    ((unsigned int&)x)

    Inside = true;
    const Vector3& MinB = box.low();
    const Vector3& MaxB = box.high();
    Vector3 MaxT(-1.0f, -1.0f, -1.0f);

    // Find candidate planes.
    for (int i = 0; i < 3; ++i) {
        if (origin[i] < MinB[i]) {
            location[i]    = MinB[i];
            Inside      = false;

            // Calculate T distances to candidate planes
            if (IR(dir[i])) {
                MaxT[i] = (MinB[i] - origin[i]) / dir[i];
            }
        } else if (origin[i] > MaxB[i]) {
            location[i]    = MaxB[i];
            Inside        = false;

            // Calculate T distances to candidate planes
            if (IR(dir[i])) {
                MaxT[i] = (MaxB[i] - origin[i]) / dir[i];
            }
        }
    }

    if (Inside) {
        // Ray origin inside bounding box
        location = origin;
        return false;
    }

    // Get largest of the maxT's for final choice of intersection
    int WhichPlane = 0;
    if (MaxT[1] > MaxT[WhichPlane])    {
        WhichPlane = 1;
    }

    if (MaxT[2] > MaxT[WhichPlane])    {
        WhichPlane = 2;
    }

    // Check final candidate actually inside box
    if (IR(MaxT[WhichPlane]) & 0x80000000) {
        // Miss the box
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        if (i != WhichPlane) {
            location[i] = origin[i] + MaxT[WhichPlane] * dir[i];
            if ((location[i] < MinB[i]) ||
                (location[i] > MaxB[i])) {
                // On this plane we're outside the box extents, so
                // we miss the box
                return false;
            }
        }
    }

    // Choose the normal to be the plane normal facing into the ray
    normal = Vector3::zero();
    normal[WhichPlane] = (dir[WhichPlane] > 0) ? -1.0 : 1.0;

    return true;

    #undef IR
}

// From G3D::CollisionDetection
inline float collisionTimeForMovingPointFixedAABox(
    const Vector3&          origin,
    const Vector3&          dir,
    const AxisAlignedBox3&  box,
    Vector3&                location,
    bool&                   Inside,
    Vector3&                normal) {

    if (collisionLocationForMovingPointFixedAABox(origin, dir, box, location, Inside, normal)) {
        return std::sqrt((location - origin).squaredMagnitude() / dir.squaredMagnitude());
    } else {
        return -1;
    }
}

inline RayIntersection3
AxisAlignedBox3::rayIntersection(Ray3 const & ray, Real max_time) const
{
  // Early exit
  if (max_time >= 0)
  {
    Real ray_sqlen = ray.getDirection().squaredLength();
    Real origin_sqdist = squaredDistance(ray.getOrigin());  // fast operation
    if (origin_sqdist > max_time * max_time * ray_sqlen)
      return RayIntersection3(-1);
  }

  Vector3 p, n;
  bool inside;
  Real t = collisionTimeForMovingPointFixedAABox(ray.getOrigin(), ray.getDirection(), *this, p, inside, n);
  if (inside)
    return RayIntersection3(0);
  else if (t >= 0 && (max_time < 0 || t <= max_time))
    return RayIntersection3(t, &n);
  else
    return RayIntersection3(-1);
}

/**
 * An arbitrarily oriented ball in 3-space.
 *
 * @note A (2-)sphere is the 2-dimensional surface of a 3-dimensional ball.
 */
class Ball3
{
  public:
    /** Default constructor. Does not initialize anything. */
    Ball3() {}

    /** Initialize with a center and a radius. */
    Ball3(Vector3 const & center_, Real radius_) : center(center_), radius(radius_) {}

    /** Copy constructor. */
    Ball3(Ball3 const & src) : center(src.center), radius(src.radius) {}

    /** Get the center of the ball. */
    Vector3 const & getCenter() const { return center; }

    /** Set the center of the ball. */
    void setCenter(Vector3 const & center_) { center = center_; }

    /** Get the radius of the ball. */
    Real getRadius() const { return radius; }

    /** Set the radius of the ball. */
    void setRadius(Real radius_) { radius = radius_; }

    /** Test if this ball intersects (i.e. contains) a point. */
    bool intersects(Vector3 const & p) const { return contains(p); }

    /** Check if the ball intersects another ball. */
    bool intersects(Ball3 const & other) const
    {
      return (center - other.center).length() < radius + other.radius;
    }

    /** Check if the ball intersects an axis-aligned box. */
    bool intersects(AxisAlignedBox3 const & aab) const
    {
      return aab.squaredDistance(center) <= radius * radius;
    }

    /** Check if the ball contains a point. */
    bool contains(Vector3 const & p) const { return (p - center).squaredLength() <= radius * radius; }

    /** Check if the ball contains another ball. */
    bool contains(Ball3 const & other) const
    {
      return radius >= other.radius && (center - other.center).length() < radius - other.radius;
    }

    /** Check if the ball contains an axis-aligned box. */
    bool contains(AxisAlignedBox3 const & aab) const
    {
      for (int i = 0; i < 8; ++i)
        if (!contains(aab.corner(i)))
          return false;

      return true;
    }

    /** Get the distance of the ball from another point. */
    Real distance(Vector3 const & p) const
    {
      return std::max((p - center).length() - radius, static_cast<Real>(0));
    }

    /** Get the distance of the ball from a point. */
    Real distance(Ball3 const & other) const
    {
      return std::max((other.center - center).length() - radius - other.radius, static_cast<Real>(0));
    }

    /** Get the distance of the ball from an axis-aligned box. */
    Real distance(AxisAlignedBox3 const & aab) const
    {
      return std::max(aab.distance(center) - radius, (Real)0);
    }

    /** Get the squared distance of the ball from a point, ball or axis-aligned box. */
    template <typename OtherType> Real squaredDistance(OtherType const & other) const
    { Real x = distance(other); return x * x; }

    /** Get the squared distance of the ball from an axis-aligned box. */
    Real squaredDistance(AxisAlignedBox3 const & aab) const { Real x = distance(aab); return x * x; }

    /** Get a bounding box for the ball. */
    AxisAlignedBox3 getBounds() const
    {
      Vector3 ext(radius, radius, radius);
      return AxisAlignedBox3(center - ext, center + ext);
    }

    /** Get a textual representation of the ball. */
    std::string toString() const
    {
      std::ostringstream oss;
      oss << "[center: " << center.toString() << ", radius: " << radius << ']';
      return oss.str();
    }

    bool rayIntersects(Ray3 const & ray, Real max_time = -1) const
    {
      if (max_time >= 0)
        return rayIntersectionTime(ray, max_time) >= 0;

      Vector3 co = ray.getOrigin() - center;

      Real c = co.squaredLength() - radius * radius;
      if (c <= 0)  // origin is inside ball
        return true;

      Real a = ray.getDirection().squaredLength();
      Real b = 2 * co.dot(ray.getDirection());

      // Solve quadratic a * t^2 + b * t + c = 0
      Real b2 = b * b;
      Real det = b2 - 4 * a * c;
      if (det < 0) return false;

      if (a > 0)
        return b <= 0 || det >= b2;
      else if (a < 0)
        return b >= 0 || det >= b2;
      else
        return false;
    }

    Real rayIntersectionTime(Ray3 const & ray, Real max_time = -1) const
    {
      Vector3 co = ray.getOrigin() - center;

      Real c = co.squaredLength() - radius * radius;
      if (c <= 0)  // origin is inside ball
        return 0;

      // We could do an early test to see if the distance from the ray origin to the ball is less than
      // max_time * ray.getDirection().length(), but it would involve a square root so might as well solve the quadratic.

      Real a = ray.getDirection().squaredLength();
      Real b = 2 * co.dot(ray.getDirection());

      // Solve quadratic a * t^2 + b * t + c = 0
      Real det = b * b - 4 * a * c;
      if (det < 0) return -1;

      Real d = std::sqrt(det);
      Real t = -1;
      if (a > 0)
      {
        Real s0 = -b - d;
        if (s0 >= 0)
          t = s0 / (2 * a);
        else
        {
          Real s1 = -b + d;
          if (s1 >= 0)
            t = s1 / (2 * a);
        }
      }
      else if (a < 0)
      {
        Real s0 = -b + d;
        if (s0 <= 0)
          t = s0 / (2 * a);
        else
        {
          Real s1 = -b - d;
          if (s1 <= 0)
            t = s1 / (2 * a);
        }
      }

      if (max_time >= 0 && t > max_time)
        return -1;
      else
        return t;
    }

    RayIntersection3 rayIntersection(Ray3 const & ray, Real max_time = -1) const
    {
      Real t = rayIntersectionTime(ray, max_time);
      if (t >= 0)
      {
        Vector3 p = ray.getPoint(t);
        Vector3 n = p - center;
        return RayIntersection3(t, &n);
      }
      else
        return RayIntersection3(-1);
    }

  private:
    Vector3  center;  ///< Center of the ball.
    Real     radius;  ///< Radius of the ball.

}; // class Ball3

/**
 An infinite 2D plane in 3D space.
 */
class Plane3 {
private:

    /** normal.Dot(x,y,z) = distance */
    Vector3						_normal;
    float						_distance;

    /**
     Assumes the normal has unit length.
     */
    Plane3(const Vector3& n, float d) : _normal(n), _distance(d) {
    }

public:

    Plane3() : _normal(Vector3::unitY()), _distance(0) {
    }

    /**
     Constructs a plane from three points.
     */
    Plane3(
        const Vector3&      point0,
        const Vector3&      point1,
        const Vector3&      point2) {

        _normal   = (point1 - point0).cross(point2 - point0).direction();
        _distance = _normal.dot(point0);
    }

    /**
     The normal will be unitized.
     */
    Plane3(
        const Vector3&      __normal,
        const Vector3&      point) {

        _normal    = __normal.direction();
        _distance  = _normal.dot(point);
    }

    static Plane3 fromEquation(float a, float b, float c, float d) {
        Vector3 n(a, b, c);
        float magnitude = n.magnitude();
        d /= magnitude;
        n /= magnitude;
        return Plane3(n, -d);
    }

    virtual ~Plane3() {}

    /**
     Returns true if point is on the side the normal points to or
     is in the plane.
     */
    inline bool halfSpaceContains(Vector3 point) const {
        // Clamp to a finite range for testing
        point = point.clamp(Vector3::minFinite(), Vector3::maxFinite());

        // We can get away with putting values *at* the limits of the float32 range into
        // a dot product, since the dot product is carried out on float64.
        return _normal.dot(point) >= _distance;
    }

    /**
     Returns true if point is on the side the normal points to or
     is in the plane.  Only call on finite points.  Faster than halfSpaceContains.
     */
    inline bool halfSpaceContainsFinite(const Vector3& point) const {
        return _normal.dot(point) >= _distance;
    }

    /**
     Returns true if the point is nearly in the plane.
     */
    inline bool fuzzyContains(const Vector3 &point) const {
        return G3D::fuzzyEq(point.dot(_normal), _distance);
    }

	inline const Vector3& normal() const {
		return _normal;
	}

    /**
      Returns distance from point to plane. Distance is negative if point is behind (not in plane in direction opposite normal) the plane.
    */
    inline float distance(const Vector3& x) const {
        return (_normal.dot(x) - _distance);
    }

    inline Vector3 closestPoint(const Vector3& x) const {
        return x + (_normal * (-distance(x)));
    }

    /** Returns normal * distance from origin */
    Vector3 center() const {
        return _normal * _distance;
    }

    /**
     Inverts the facing direction of the plane so the new normal
     is the inverse of the old normal.
     */
    void flip() {
        _normal   = -_normal;
        _distance  = -_distance;
    }

    /**
      Returns the equation in the form:

      <CODE>normal.Dot(Vector3(<I>x</I>, <I>y</I>, <I>z</I>)) + d = 0</CODE>
     */
    void getEquation(Vector3 &normal, double& d) const {
        double _d;
        getEquation(normal, _d);
        d = (float)_d;
    }

    void getEquation(Vector3 &normal, float& d) const {
        normal = _normal;
        d = -_distance;
    }

    /**
      ax + by + cz + d = 0
     */
    void getEquation(double& a, double& b, double& c, double& d) const {
        double _a, _b, _c, _d;
        getEquation(_a, _b, _c, _d);
        a = (float)_a;
        b = (float)_b;
        c = (float)_c;
        d = (float)_d;
    }

    void getEquation(float& a, float& b, float& c, float& d) const {
        a = _normal.x;
        b = _normal.y;
        c = _normal.z;
        d = -_distance;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Plane3(" << _normal.x << ", " << _normal.y << ", " << _normal.z << ", " << _distance << ')';
        return oss.str();
    }

}; // struct Plane3

/**
 A finite segment of an infinite 3D line.
 */
class LineSegment3 {
protected:

    Vector3             _point;

    /** Not normalized */
    Vector3             direction;

    LineSegment3(const Vector3& __point, const Vector3& _direction) : _point(__point), direction(_direction) {
    }

public:

    inline LineSegment3() : _point(Vector3::zero()), direction(Vector3::zero()) {}

    virtual ~LineSegment3() {}

    /**
     * Constructs a line from two (not equal) points.
     */
    static LineSegment3 fromTwoPoints(const Vector3 &point1, const Vector3 &point2) {
        return LineSegment3(point1, point2 - point1);
    }

	/** Call with 0 or 1 */
    Vector3 point(int i) const {
        switch (i) {
        case 0:
            return _point;

        case 1:
            return _point + direction;

        default:
            THEA_ASSERT(i == 0 || i == 1, "Argument to point must be 0 or 1");
            return _point;
        }
    }

    inline float length() const {
        return direction.magnitude();
    }

    /**
     * Returns the closest point on the line segment to point.
     */
    Vector3 closestPoint(const Vector3 &p) const {
        // The vector from the end of the capsule to the point in question.
        Vector3 v(p - _point);

        // Projection of v onto the line segment scaled by
        // the length of direction.
        float t = direction.dot(v);

        // Avoid some square roots.  Derivation:
        //    t/direction.length() <= direction.length()
        //      t <= direction.squaredLength()

        if ((t >= 0) && (t <= direction.squaredMagnitude())) {

            // The point falls within the segment.  Normalize direction,
            // divide t by the length of direction.
            return _point + direction * t / direction.squaredMagnitude();

        } else {

            // The point does not fall within the segment; see which end is closer.

            // Distance from 0, squared
            float d0Squared = v.squaredMagnitude();

            // Distance from 1, squared
            float d1Squared = (v - direction).squaredMagnitude();

            if (d0Squared < d1Squared) {

                // Point 0 is closer
                return _point;

            } else {

                // Point 1 is closer
                return _point + direction;

            }
        }
    }

    /**
     Returns the distance between point and the line
     */
    double distance(const Vector3& p) const {
        return (closestPoint(p) - p).magnitude();
    }

    double distanceSquared(const Vector3& p) const {
        return (closestPoint(p) - p).squaredMagnitude();
    }

};

/**
 * Stores the positions of a triangle's three vertices locally, and provides access to them.
 *
 * @see Triangle3
 */
class TriangleLocalVertexTriple3
{
  public:
    /** Default constructor. */
    TriangleLocalVertexTriple3() {}

    /** Initializing constructor. */
    TriangleLocalVertexTriple3(Vector3 const & v0, Vector3 const & v1, Vector3 const & v2)
    {
      vertices[0] = v0;
      vertices[1] = v1;
      vertices[2] = v2;
    }

    /** Get the i'th vertex. */
    Vector3 const & getVertex(int i) const { return vertices[i]; }

  private:
    Vector3 vertices[3];  ///< Vertex positions.

}; // TriangleLocalVertexTriple3

// Internal functions
namespace Triangle3Internal {

// Get the closest pair of points between two line segments, and the square of the distance between them. From Christer Ericson,
// "Real-Time Collision Detection", Morgan-Kaufman, 2005.
Real closestPtSegmentSegment(Vector3 const & p1, Vector3 const & q1, Vector3 const & p2, Vector3 const & q2, Real & s, Real & t,
                             Vector3 & c1, Vector3 & c2);

/* Triangle/triangle intersection test routine,
 * by Tomas Moller, 1997.
 * See article "A Fast Triangle-Triangle Intersection Test",
 * Journal of Graphics Tools, 2(2), 1997
 * updated: 2001-06-20 (added line of intersection)
 *
 * int tri_tri_intersect(Real const V0[3],Real const V1[3],Real const V2[3],
 *                       Real const U0[3],Real const U1[3],Real const U2[3])
 *
 * parameters: vertices of triangle 1: V0,V1,V2
 *             vertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 *
 * Here is a version withouts divisions (a little faster)
 * int NoDivTriTriIsect(Real const V0[3],Real const V1[3],Real const V2[3],
 *                      Real const U0[3],Real const U1[3],Real const U2[3]);
 *
 * This version computes the line of intersection as well (if they are not coplanar):
 * int tri_tri_intersect_with_isectline(Real const V0[3],Real const V1[3],Real const V2[3],
 *                                      Real const U0[3],Real const U1[3],Real const U2[3],int *coplanar,
 *                                      Real isectpt1[3],Real isectpt2[3]);
 * coplanar returns whether the tris are coplanar
 * isectpt1, isectpt2 are the endpoints of the line of intersection
 */

int tri_tri_intersect(Real const V0[3],Real const V1[3],Real const V2[3], Real const U0[3],Real const U1[3],Real const U2[3]);

int NoDivTriTriIsect(Real const V0[3],Real const V1[3],Real const V2[3], Real const U0[3],Real const U1[3],Real const U2[3]);

int tri_tri_intersect_with_isectline(Real const V0[3],Real const V1[3],Real const V2[3],
                                     Real const U0[3],Real const U1[3],Real const U2[3],int *coplanar,
                                     Real isectpt1[3],Real isectpt2[3]);

// Intersection time of a ray with a triangle. Returns a negative value if the ray does not intersect the triangle.
Real rayTriangleIntersectionTime(Ray3 const & ray, Vector3 const & v0, Vector3 const & edge01, Vector3 const & edge02);

// From G3D::CollisionDetection
bool isPointInsideTriangle(const Vector3 & v0, const Vector3 & v1, const Vector3 & v2, const Vector3 & normal,
                           const Vector3 & point, float b[3], Vector3::Axis primaryAxis);

inline bool isPointInsideTriangle(
    const Vector3&			v0,
    const Vector3&			v1,
    const Vector3&			v2,
    const Vector3&			normal,
    const Vector3&			point,
    Vector3::Axis           primaryAxis = Vector3::DETECT_AXIS) {

    float b[3];
    return isPointInsideTriangle(v0, v1, v2, normal, point, b, primaryAxis);
}

Vector3 closestPointOnLineSegment(
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& point);

Vector3 closestPointOnLineSegment(
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& edgeDirection,
    const float    edgeLength,
    const Vector3& point);

Vector3 closestPointOnTrianglePerimeter(
    const Vector3&			v0,
    const Vector3&			v1,
    const Vector3&			v2,
    const Vector3&			point);

Vector3 closestPointOnTrianglePerimeter(
    const Vector3   v[3],
    const Vector3   edgeDirection[3],
    const float     edgeLength[3],
    const Vector3&  point,
    int&            edgeIndex);

} // namespace Triangle3Internal

/**
 * Base class for a triangle in 3-space, with precomputed properties for fast access. To account for the fact that the triangle
 * vertices may be stored in different ways (e.g. internally within the object, or as indices into an external vertex pool), the
 * class is parametrized on the way the vertices are stored. <code>VertexTripleT</code> must be default-constructible and
 * provide an efficient member function with the signature
 *
 * \code
 * Vector3 const & getVertex(int i) const
 * \endcode
 *
 * @note: This class cannot be used directly (it has protected constructors). Use Triangle3 or LocalTriangle3 instead.
 */
template <typename VertexTripleT = TriangleLocalVertexTriple3>
class Triangle3Base
{
  public:
    typedef VertexTripleT VertexTriple;  ///< Stores and provides access to the triangle's vertices.

  protected:
    /** Default constructor. Does not initialize anything. */
    Triangle3Base() {}

    /** Construct from a set of vertices. */
    explicit Triangle3Base(VertexTriple const & vertices_) : vertices(vertices_) { update(); }

    /** Copy constructor. */
    Triangle3Base(Triangle3Base const & src)
    : vertices(src.vertices), plane(src.plane), primary_axis(src.primary_axis), centroid(src.centroid), edge01(src.edge01),
      edge02(src.edge02), area(src.area)
    {}

  public:
    /** Initialize the triangle from its vertices. */
    void set(VertexTriple const & vertices_) { vertices = vertices_; update(); }

    /**
     * Update the properties of the triangle, assuming the positions of its three corners have changed. Useful for external
     * callers if the mutable positions are not locally stored within the triangle.
     */
    void update()
    {
      plane = Plane3(getVertex(0), getVertex(1), getVertex(2));
      primary_axis = CoordinateAxis(plane.normal().primaryAxis());

      centroid = (getVertex(0) + getVertex(1) + getVertex(2)) / 3;
      edge01   = getVertex(1) - getVertex(0);
      edge02   = getVertex(2) - getVertex(0);

      area = 0.5f * edge01.cross(edge02).length();
    }

    /** Get a vertex of the triangle. */
    Vector3 const & getVertex(int i) const { return vertices.getVertex(i); }

    /** Get the vertices of the triangle. */
    VertexTriple const & getVertices() const { return vertices; }

    /** Get the plane of the triangle. */
    Plane3 const & getPlane() const { return plane; }

    /** Get the primary axis of the triangle (closest to normal). */
    CoordinateAxis getPrimaryAxis() const { return primary_axis; }

    /** Get the normal of the triangle (right-hand rule, going round vertices in order 0, 1, 2). */
    Vector3 const & getNormal() const { return plane.normal(); }

    /** Get the centroid of the triangle. */
    Vector3 const & getCentroid() const { return centroid; }

    /** Get the edge vector corresponding to getVertex(1) - getVertex(0). */
    Vector3 const & getEdge01() const { return edge01; }

    /** Get the edge vector corresponding to getVertex(2) - getVertex(0). */
    Vector3 const & getEdge02() const { return edge02; }

    /** Get the area of the triangle. */
    Real getArea() const { return area; }

    /** Get a uniformly distributed random sample from the triangle. */
    Vector3 randomPoint() const
    {
      // From G3D::Triangle

      // Choose a random point in the parallelogram
      float s = std::rand() / (float)RAND_MAX;
      float t = std::rand() / (float)RAND_MAX;

      if (s + t > 1.0f)
      {
        // Outside the triangle; reflect about the diagonal of the parallelogram
        t = 1.0f - t;
        s = 1.0f - s;
      }

      return s * edge01 + t * edge02 + getVertex(0);
    }

    /** Get a bounding box for the triangle. */
    AxisAlignedBox3 getBounds() const
    {
      return AxisAlignedBox3(getVertex(0).min(getVertex(1).min(getVertex(2))),
                             getVertex(0).max(getVertex(1).max(getVertex(2))));
    }

    /** Check if the triangle intersects (that is, contains) a point. */
    bool intersects(Vector3 const & p) const { return contains(p); }

    /** Check if the triangle intersects another triangle. */
    template <typename OtherVertexTripleT> bool intersects(Triangle3Base<OtherVertexTripleT> const & other) const
    {
      Vector3 const & p0 = getVertex(0);       Vector3 const & p1 = getVertex(1);       Vector3 const & p2 = getVertex(2);
      Vector3 const & q0 = other.getVertex(0); Vector3 const & q1 = other.getVertex(1); Vector3 const & q2 = other.getVertex(2);

      Real v0[3] = { p0.x, p0.y, p0.z};
      Real v1[3] = { p1.x, p1.y, p1.z};
      Real v2[3] = { p2.x, p2.y, p2.z};

      Real u0[3] = { q0.x, q0.y, q0.z};
      Real u1[3] = { q1.x, q1.y, q1.z};
      Real u2[3] = { q2.x, q2.y, q2.z};

      return Triangle3Internal::NoDivTriTriIsect(v0, v1, v2, u0, u1, u2);
    }

    /**
     * Check if the triangle intersects another triangle. If they do intersect, check if they are coplanar. If they are not
     * coplanar, compute the line of intersection. The intersection test is somewhat slower than intersects().
     */
    template <typename OtherVertexTripleT>
    bool intersects(Triangle3Base<OtherVertexTripleT> const & other, bool & coplanar, LineSegment3 & seg) const
    {
      Vector3 const & p0 = getVertex(0);       Vector3 const & p1 = getVertex(1);       Vector3 const & p2 = getVertex(2);
      Vector3 const & q0 = other.getVertex(0); Vector3 const & q1 = other.getVertex(1); Vector3 const & q2 = other.getVertex(2);

      Real v0[3] = { p0.x, p0.y, p0.z};
      Real v1[3] = { p1.x, p1.y, p1.z};
      Real v2[3] = { p2.x, p2.y, p2.z};

      Real u0[3] = { q0.x, q0.y, q0.z};
      Real u1[3] = { q1.x, q1.y, q1.z};
      Real u2[3] = { q2.x, q2.y, q2.z};

      int i_coplanar;
      Real isectpt1[3], isectpt2[3];
      int isec = Triangle3Internal::tri_tri_intersect_with_isectline(v0, v1, v2, u0, u1, u2, &i_coplanar, isectpt1, isectpt2);
      if (isec)
      {
        coplanar = (bool)i_coplanar;
        if (!coplanar)
          seg = LineSegment3::fromTwoPoints(Vector3(isectpt1[0], isectpt1[1], isectpt1[2]),
                                            Vector3(isectpt2[0], isectpt2[1], isectpt2[2]));

        return true;
      }

      return false;
    }

    /** Check if the triangle intersects a ball. */
    bool intersects(Ball3 const & ball) const
    { throw std::runtime_error("Triangle3: Intersection with ball not implemented"); }

    /** Check if the triangle intersects an axis-aligned box. */
    bool intersects(AxisAlignedBox3 const & aab) const
    { throw std::runtime_error("Triangle3: Intersection with AAB not implemented"); }

    /** Check if the triangle contains a point. */
    bool contains(Vector3 const & p) const
    {
      return Triangle3Internal::isPointInsideTriangle(getVertex(0), getVertex(1), getVertex(2), getNormal(), p,
                                                      static_cast<G3D::Vector3::Axis>(getPrimaryAxis()));
    }

    /** Get the distance of the triangle from a point. */
    Real distance(Vector3 const & p) const { return std::sqrt(squaredDistance(p)); }

    /** Get the distance of the triangle from another triangle. */
    template <typename OtherVertexTripleT>
    Real distance(Triangle3Base<OtherVertexTripleT> const & other) const { return std::sqrt(squaredDistance(other)); }

    /** Get the distance of the triangle from a ball. */
    Real distance(Ball3 const & ball) const
    {
      return std::max(distance(ball.getCenter()) - ball.getRadius(), static_cast<Real>(0));
    }

    /** Get the squared distance of the triangle from a point. */
    Real squaredDistance(Vector3 const & p) const
    {
      return (closestPoint(p) - p).squaredLength();
    }

    /** Get the point on this triangle closest to a given point. */
    Vector3 closestPoint(Vector3 const & p) const
    {
      // Project the point onto the plane of the triangle
      Vector3 proj = plane.closestPoint(p);

      if (contains(proj))
        return proj;
      else  // the closest point is on the perimeter instead
        return Triangle3Internal::closestPointOnTrianglePerimeter(getVertex(0), getVertex(1), getVertex(2), p);
    }

    /** Get the squared distance of the triangle from another triangle. */
    template <typename OtherVertexTripleT> Real squaredDistance(Triangle3Base<OtherVertexTripleT> const & other) const
    {
      Vector3 p, q;
      return closestPoints(other, p, q);
    }

    /** Get the closest pair of points between two triangles, and the square of the distance between them. */
    template <typename OtherVertexTripleT>
    Real closestPoints(Triangle3Base<OtherVertexTripleT> const & other, Vector3 & this_pt, Vector3 & other_pt) const
    {
      // From Christer Ericson, "Real-Time Collision Detection", Morgan-Kaufman, 2005.

      // First test for intersection
      bool coplanar;
      LineSegment3 seg;
      if (intersects(other, coplanar, seg))
      {
        if (coplanar)
          this_pt = other_pt = 0.5 * (getCentroid() + other.getCentroid());  // FIXME: This need not be in the intersection
        else
          this_pt = other_pt = 0.5 * (seg.point(0) + seg.point(1));

        return 0;
      }

      Real min_sqdist = std::numeric_limits<Real>::max();

      // Edge-edge distances
      Vector3 p, q;
      Real d, s, t;
      for (int i = 0; i < 3; ++i)
      {
        int i2 = (i + 1) % 3;

        for (int j = 0; j < 3; ++j)
        {
          int j2 = (j + 1) % 3;
          d = Triangle3Internal::closestPtSegmentSegment(getVertex(i), getVertex(i2), other.getVertex(j), other.getVertex(j2),
                                                         s, t, p, q);
          if (d < min_sqdist)
          {
            min_sqdist = d;
            this_pt = p;
            other_pt = q;
          }
        }
      }

      // Distance from vertex of triangle 2 to triangle 1, if the former projects inside the latter
      for (int i = 0; i < 3; ++i)
      {
        q = other.getVertex(i);
        p = getPlane().closestPoint(q);
        if (Triangle3Internal::isPointInsideTriangle(getVertex(0), getVertex(1), getVertex(2), getNormal(), p,
                                                     static_cast<G3D::Vector3::Axis>(getPrimaryAxis())))
        {
          d = (p - q).squaredLength();
          if (d < min_sqdist)
          {
            min_sqdist = d;
            this_pt = p;
            other_pt = q;
          }
        }
      }

      // Distance from vertex of triangle 1 to triangle 2, if the former projects inside the latter
      for (int i = 0; i < 3; ++i)
      {
        p = getVertex(i);
        q = other.getPlane().closestPoint(p);
        if (Triangle3Internal::isPointInsideTriangle(other.getVertex(0), other.getVertex(1), other.getVertex(2),
                                                     other.getNormal(), q,
                                                     static_cast<G3D::Vector3::Axis>(other.getPrimaryAxis())))
        {
          d = (p - q).squaredLength();
          if (d < min_sqdist)
          {
            min_sqdist = d;
            this_pt = p;
            other_pt = q;
          }
        }
      }

      return d;
    }

    /** Get the squared distance of the triangle from a ball. */
    Real squaredDistance(Ball3 const & ball) const { Real x = distance(ball); return x * x; }

    /**
     * Get the point on this triangle and the point on a ball closest to each other, and return the squared distance between
     * them.
     */
    Real closestPoints(Ball3 const & ball, Vector3 & this_pt, Vector3 & ball_pt) const
    {
      this_pt = closestPoint(ball.getCenter());

      Vector3 diff = this_pt - ball.getCenter();
      Real d2 = diff.squaredLength();
      Real r2 = ball.getRadius() * ball.getRadius();
      if (d2 < r2)  // point inside ball
      {
        ball_pt = this_pt;
        return 0;
      }
      else
      {
        if (r2 < 1e-30)
        {
          ball_pt = ball.getCenter();
          return d2;
        }
        else
        {
          ball_pt = ball.getCenter() + std::sqrt(r2 / d2) * diff;
          return (this_pt - ball_pt).squaredLength();
        }
      }
    }

    Real rayIntersectionTime(Ray3 const & ray, Real max_time = -1) const
    {
      Real t = Triangle3Internal::rayTriangleIntersectionTime(ray, getVertex(0), getEdge01(), getEdge02());
      return (max_time >= 0 && t > max_time) ? -1 : t;
    }

    RayIntersection3 rayIntersection(Ray3 const & ray, Real max_time = -1) const
    {
      Real t = Triangle3Internal::rayTriangleIntersectionTime(ray, getVertex(0), getEdge01(), getEdge02());
      if (t >= 0 && (max_time < 0 || t <= max_time))
      {
        Vector3 n = getNormal();
        return RayIntersection3(t, &n);
      }

      return RayIntersection3(-1);
    }

  protected:
    VertexTriple    vertices;      ///< The vertices of the triangle.
    Plane3          plane;         ///< Plane of the triangle.
    CoordinateAxis  primary_axis;  ///< Primary axis (closest to normal).
    Vector3         centroid;      ///< Centroid of the triangle (mean of three vertices).
    Vector3         edge01;        ///< vertices[1] - vertices[0]
    Vector3         edge02;        ///< vertices[2] - vertices[0]
    Real            area;          ///< Triangle area.

}; // class Triangle3Base

// Forward declaration
template <typename VertexTripleT = TriangleLocalVertexTriple3> class Triangle3;

/**
 * A triangle with three vertex positions stored locally, in the class itself. This class adds a more direct constructor and
 * set() method for convenience to the default Triangle3 template.
 *
 * @see Triangle3
 */
template <>
class Triangle3<TriangleLocalVertexTriple3> : public Triangle3Base<TriangleLocalVertexTriple3>
{
  private:
    typedef Triangle3Base<TriangleLocalVertexTriple3> BaseT;

  public:
    /** Default constructor. Does not initialize anything. */
    Triangle3() {}

    /** Construct from a set of vertices. */
    explicit Triangle3(TriangleLocalVertexTriple3 const & vertices_) : BaseT(vertices_) {}

    /** Construct from a set of vertices. */
    Triangle3(Vector3 const & v0, Vector3 const & v1, Vector3 const & v2) : BaseT(TriangleLocalVertexTriple3(v0, v1, v2)) {}

    /** Copy constructor. */
    Triangle3(Triangle3 const & src) : BaseT(src) {}

    /** Initialize the triangle from its vertices. */
    void set(Vector3 const & v0, Vector3 const & v1, Vector3 const & v2)
    {
      BaseT::set(TriangleLocalVertexTriple3(v0, v1, v2));
      update();
    }

    /** Get a copy of this triangle (for consistency with the default template). */
    Triangle3 localClone() const { return *this; }

}; // class Triangle3<TriangleLocalVertexTriple3>

/** A triangle with three vertex positions stored locally, in the class itself. */
typedef Triangle3<TriangleLocalVertexTriple3> LocalTriangle3;

/**
 * A triangle in 3-space, with precomputed properties for fast access. To account for the fact that the triangle vertices may
 * be stored in different ways (e.g. internally within the object, or as indices into an external vertex pool), the class is
 * parametrized on the way the vertices are stored. <code>VertexTripleT</code> must be default-constructible and provide an
 * efficient member function with the signature
 *
 * \code
 * Vector3 const & getVertex(int i) const
 * \endcode
 */
template <typename VertexTripleT>
class Triangle3 : public Triangle3Base<VertexTripleT>
{
  private:
    typedef Triangle3Base<VertexTripleT> BaseT;

  public:
    /** Default constructor. Does not initialize anything. */
    Triangle3() {}

    /** Construct from a set of vertices. */
    explicit Triangle3(VertexTripleT const & vertices_) : BaseT(vertices_) {}

    /** Copy constructor. */
    Triangle3(Triangle3 const & src) : BaseT(src) {}

    /** Get a new triangle that simply stores copies of the vertex positions of this triangle. */
    LocalTriangle3 localClone() const
    {
      return LocalTriangle3(BaseT::getVertex(0), BaseT::getVertex(1), BaseT::getVertex(2));
    }

}; // class Triangle3

/**
 * Interface for all object filters. A filter has an allows() function that takes an object argument and returns a boolean value
 * indicating whether the filter allows the object to pass through or not.
 */
template <typename T>
class Filter
{
  public:
    /** Destructor. */
    virtual ~Filter() {}

    /**
     * Check if the filter allows an object to pass through or not.
     *
     * @return True if the object \a t is allowed through, false if it is blocked.
     */
    virtual bool allows(T const & t) const = 0;

}; // class Filter

/** A filter that allows everything to pass through. */
template <typename T>
class AlwaysPassFilter : public Filter<T>
{
  public:
    bool allows(T const & t) const { return true; }

}; // class AlwaysPassFilter

/** A filter that allows nothing to pass through. */
template <typename T>
class AlwaysBlockFilter : public Filter<T>
{
  public:
    bool allows(T const & t) const { return false; }

}; // class AlwaysPassFilter

/**
 * A base class for objects that should never be copied. This is achieved by declaring the copy constructor and assignment
 * operator as private members. <b>Never ever</b> try to refer to an object of a derived class using a Noncopyable pointer or
 * reference (in any case this seems semantically weird) -- to ensure this class has zero runtime overhead, the destructor is
 * <b>not virtual</b>.
 */
class Noncopyable
{
  protected:
    /** Constructor. */
    Noncopyable() {}

    /** Destructor. */
    ~Noncopyable() {}

  private:
    /**
     * Hidden copy constructor. No body provided since this should never be accessible -- if a linker error occurs then
     * something is seriously wrong.
     */
    Noncopyable(const Noncopyable &);

    /**
     * Hidden assignment operator. No body provided since this should never be accessible -- if a linker error occurs then
     * something is seriously wrong.
     */
    Noncopyable const & operator=(Noncopyable const &);

}; // class Noncopyable

/**
 * Has boolean member <code>value = true</code> if <code>T</code> can be identified with a single point in 3D space, else
 * <code>value = false</code>. Unless you specialize the class to set the value to true, it is false by default.
 *
 * @see PointTraits3
 */
template <class T>
struct IsPoint3
{
  static bool const value = false;
};

// Specialization for Vector3
template <>
struct IsPoint3<Vector3>
{
  static bool const value = true;
};

/**
 * Traits for an object which can be identified with a single point in 3-space.
 *
 * @see IsPoint3
 */
template <typename T>
struct PointTraits3
{
  /** Return the position of the point. Specialize as required. */
  static Vector3 const & getPosition(T const & t) { return t; }
};

// Partial specialization of PointTraits3 for pointer types
template <class T>
struct PointTraits3<T *>
{
  static Vector3 const & getPosition(T const * t) { return PointTraits3<T>::getPosition(*t); }
};

/** Traits class for a bounded object in 3-space. */
template <typename T, typename Enable = void>
class BoundedObjectTraits3
{
  public:
    /** Get a bounding range for an object. */
    template <typename RangeT> static void getBounds(T const & t, RangeT & bounds) { bounds = t.getBounds(); }

    /** Get the center of the object. */
    static Vector3 getCenter(T const & t) { return t.getBounds().center(); }

    /** Get the maximum position of the object along a particular coordinate axis. */
    static Real getHigh(T const & t, int coord) { return t.getBounds().high()[coord]; }

    /** Get the minimum position of the object along a particular coordinate axis. */
    static Real getLow(T const & t, int coord) { return t.getBounds().low()[coord]; }

}; // class BoundedObjectTraits3

// Specialization for pointer types
template <typename T>
struct BoundedObjectTraits3<T *>
{
  template <typename RangeT> static void getBounds(T const * t, RangeT & bounds)
  {
    THEA_ASSERT(t, "BoundedObjectTraits3: Can't get bounds of null object");
    BoundedObjectTraits3<T>::getBounds(*t, bounds);
  }

  static Vector3 getCenter(T const * t)
  {
    THEA_ASSERT(t, "BoundedObjectTraits3: Can't get center of null object");
    return BoundedObjectTraits3<T>::getCenter(*t);
  }

  static Real getHigh(T const * t, int coord)
  {
    THEA_ASSERT(t, "BoundedObjectTraits3: Can't get bounds of null object");
    return BoundedObjectTraits3<T>::getHigh(*t, coord);
  }

  static Real getLow(T const * t, int coord)
  {
    THEA_ASSERT(t, "BoundedObjectTraits3: Can't get bounds of null object");
    return BoundedObjectTraits3<T>::getLow(*t, coord);
  }
};

// Specialization for 3D points
template <typename T>
struct BoundedObjectTraits3<T, typename boost::enable_if< IsPoint3<T> >::type>
{
  static void getBounds(T const & t, AxisAlignedBox3 & bounds)
  {
    Vector3 const & pos = PointTraits3<T>::getPosition(t);
    bounds.set(pos, pos);
  }

  static void getBounds(T const & t, Ball3 & bounds) { bounds = Ball3(PointTraits3<T>::getPosition(t), 0); }
  static Vector3 getCenter(T const & t) { return PointTraits3<T>::getPosition(t); }
  static Real getHigh(T const & t, int coord) { return PointTraits3<T>::getPosition(t)[coord]; }
  static Real getLow(T const & t, int coord)  { return PointTraits3<T>::getPosition(t)[coord];  }
};

// Specialization for AxisAlignedBox3
template <>
struct BoundedObjectTraits3<AxisAlignedBox3>
{
  static void getBounds(AxisAlignedBox3 const & t, AxisAlignedBox3 & bounds) { bounds = t; }

  static void getBounds(AxisAlignedBox3 const & t, Ball3 & bounds)
  { bounds = Ball3(t.center(), 0.5f * t.extent().length()); }

  static Vector3 getCenter(AxisAlignedBox3 const & t) { return t.center(); }
  static Real getHigh(AxisAlignedBox3 const & t, int coord) { return t.high()[coord]; }
  static Real getLow(AxisAlignedBox3 const & t, int coord)  { return t.low()[coord];  }
};

// Specialization for Ball3
template <>
struct BoundedObjectTraits3<Ball3>
{
  static void getBounds(Ball3 const & t, AxisAlignedBox3 & bounds) { bounds = t.getBounds(); }
  static void getBounds(Ball3 const & t, Ball3 & bounds)           { bounds = t; }

  static Vector3 getCenter(Ball3 const & t) { return t.getCenter(); }
  static Real getHigh(Ball3 const & t, int coord) { return t.getCenter()[coord] + t.getRadius(); }
  static Real getLow(Ball3 const & t, int coord)  { return t.getCenter()[coord] - t.getRadius(); }
};

// Specialization for Triangle3
template <typename VertexTripleT>
struct BoundedObjectTraits3< Triangle3<VertexTripleT> >
{
  typedef Triangle3<VertexTripleT> Triangle;

  static void getBounds(Triangle const & t, AxisAlignedBox3 & bounds) { bounds = t.getBounds(); }

  // TODO: Make this tighter
  static void getBounds(Triangle const & t, Ball3 & bounds)
  { BoundedObjectTraits3<AxisAlignedBox3>::getBounds(t.getBounds(), bounds); }

  static Vector3 getCenter(Triangle const & t) { return (t.getVertex(0) + t.getVertex(1) + t.getVertex(2)) / 3.0f; }

  static Real high(Triangle const & t, int coord)
  { return std::max(std::max(t.getVertex(0)[coord], t.getVertex(1)[coord]), t.getVertex(2)[coord]); }

  static Real low(Triangle const & t, int coord)
  { return std::min(std::min(t.getVertex(0)[coord], t.getVertex(1)[coord]), t.getVertex(2)[coord]); }
};

/**
 * Helper class for MetricL2. Specializations of this class actually compute the metric. This is required because C++ does
 * unexpected things with specialized and overloaded function templates (see http://www.gotw.ca/publications/mill17.htm).
 *
 * @see MetricL2
 *
 * @todo Complete closestPoints() specializations for all supported types.
 */
template <typename A, typename B, typename Enable = void>
struct MetricL2Impl
{
  private:
    typedef char UnspecifiedPointT;

  public:
    /** Measure the Euclidean (L2) distance between two objects. */
    static double distance(A const & a, B const & b);

    /**
     * Measure the square of the Euclidean (L2) distance between two objects, which is an efficiently computable (avoids the
     * square root) monotonic approximation to the true distance.
     */
    static double monotoneApproxDistance(A const & a, B const & b);

    /**
     * Find the closest pair of points between two objects.
     *
     * @return A monotonic approximation (the square of the Euclidean distance in this case) to the shortest distance between
     * the objects, i.e. the value of monotoneApproxDistance(\a a, \a b).
     */
    static double closestPoints(A const & a, B const & b, UnspecifiedPointT & cpa, UnspecifiedPointT & cpb);

}; // class MetricL2Impl

/**
 * Distances and closest pairs of points between various types of objects according to the Euclidean (L2) metric. When distances
 * will only be compared to one another to see which is larger, a monotonic function of the true distance works just as well and
 * may be more efficient to compute. In this case, the square of the Euclidean distance avoids a costly square root operation.
 * For this reason, it is better to use the monotoneApproxDistance() function instead of distance() wherever possible. The
 * functions computeMonotoneApprox() and invertMonotoneApprox() switch between the true and approximate distances.
 *
 * To add support for distances between custom types, add specializations (full or partial) of the helper class MetricL2Impl.
 * This class is required because C++ does unexpected things with specialized and overloaded function templates (see
 * http://www.gotw.ca/publications/mill17.htm). Note that to commutatively support distances between distinct types A and B, you
 * must specialize both MetricL2Impl<A, B> and MetricL2Impl<B, A>. Do <b>not</b> try specializing MetricL2::distance() and
 * similar functions.
 *
 * MetricL2 defines the standard interface for a metric -- its interface must be supported by all other metric classes.
 *
 * @see MetricL2Impl
 */
class MetricL2  // static members defined in header, so can't use dllimport
{
  public:
    /** Compute a fixed monotone approximation (here, square) of a distance. */
    static double computeMonotoneApprox(double d) { return d * d; }

    /** Invert the fixed monotone approximation of a distance to get the true distance (here, by taking the square root). */
    static double invertMonotoneApprox(double d) { return std::sqrt(d); }

    /**
     * Measure the Euclidean (L2) distance between two objects via the helper class MetricL2Impl. Add specializations of
     * MetricL2Impl as required for specific types of objects. Note that if types A and B differ, you must explicitly specialize
     * both MetricL2Impl<A, B> and MetricL2Impl<B, A>.
     */
    template <typename A, typename B> static double distance(A const & a, B const & b)
    { return MetricL2Impl<A, B>::distance(a, b); }

    /**
     * Measure the square of the Euclidean (L2) distance between two objects, which is an efficiently computable (avoids the
     * square root) monotonic approximation to the true distance. Add specializations of the helper class MetricL2Impl as
     * required for specific types of objects. Note that if types A and B differ, you must explicitly specialize
     * both MetricL2Impl<A, B> and MetricL2Impl<B, A>.
     */
    template <typename A, typename B> static double monotoneApproxDistance(A const & a, B const & b)
    { return MetricL2Impl<A, B>::monotoneApproxDistance(a, b); }

    /**
     * Find the closest pair of points between two objects, via the helper class MetricL2Impl. Add specializations of
     * MetricL2Impl s required for specific types of objects. Note that if types A and B differ, you must explicitly specialize
     * both MetricL2Impl<A, B> and MetricL2Impl<B, A>.
     *
     * @return A monotonic approximation (the square of the Euclidean distance in this case) to the shortest distance between
     * the objects, i.e. the value of monotoneApproxDistance(\a a, \a b).
     */
    template <typename A, typename B, typename PointT>
    static double closestPoints(A const & a, B const & b, PointT & cpa, PointT & cpb)
    { return MetricL2Impl<A, B>::closestPoints(a, b, cpa, cpb); }

}; // class MetricL2

// Support for pointer types
template <typename A, typename B>
struct MetricL2Impl<A *, B *>
{
  public:
    static double distance(A const * a, B const * b) { return MetricL2::distance(*a, *b); }
    static double monotoneApproxDistance(A const * a, B const * b) { return MetricL2::monotoneApproxDistance(*a, *b); }
    template <typename PointT> static double closestPoints(A const * a, B const * b, PointT & cpa, PointT & cpb)
    { return MetricL2::closestPoints(*a, *b, cpa, cpb); }
};

template <typename A, typename B>
struct MetricL2Impl<A, B *>
{
  public:
    static double distance(A const & a, B const * b) { return MetricL2::distance(a, *b); }
    static double monotoneApproxDistance(A const & a, B const * b) { return MetricL2::monotoneApproxDistance(a, *b); }
    template <typename PointT> static double closestPoints(A const & a, B const * b, PointT & cpa, PointT & cpb)
    { return MetricL2::closestPoints(a, *b, cpa, cpb); }
};

template <typename A, typename B>
struct MetricL2Impl<A *, B>
{
  public:
    static double distance(A const * a, B const & b) { return MetricL2::distance(*a, b); }
    static double monotoneApproxDistance(A const * a, B const & b) { return MetricL2::monotoneApproxDistance(*a, b); }
    template <typename PointT> static double closestPoints(A const * a, B const & b, PointT & cpa, PointT & cpb)
    { return MetricL2::closestPoints(*a, b, cpa, cpb); }
};

// Default specializations
template <>
struct MetricL2Impl<float, float>
{
  static double distance(float a, float b) { return std::fabs(a - b); }
  static double monotoneApproxDistance(float a, float b) { float x = a - b; return x * x; }

  static double closestPoints(float a, float b, float & cpa, float & cpb)
  { cpa = a; cpb = b; return monotoneApproxDistance(a, b); }
};

template <>
struct MetricL2Impl<double, double>
{
  static double distance(double a, double b) { return std::fabs(a - b); }
  static double monotoneApproxDistance(double a, double b) { double x = a - b; return x * x; }

  static double closestPoints(double a, double b, double & cpa, double & cpb)
  { cpa = a; cpb = b; return monotoneApproxDistance(a, b); }
};

template <typename A, typename B>
struct MetricL2Impl<A, B, typename boost::enable_if_c< IsPoint3<A>::value && IsPoint3<B>::value >::type>
{
  static double distance(A const & a, B const & b)
  { return (PointTraits3<A>::getPosition(a) - PointTraits3<B>::getPosition(b)).length(); }

  static double monotoneApproxDistance(A const & a, B const & b)
  { return (PointTraits3<A>::getPosition(a) - PointTraits3<B>::getPosition(b)).squaredLength(); }

  static double closestPoints(A const & a, B const & b, Vector3 & cpa, Vector3 & cpb)
  {
    cpa = PointTraits3<A>::getPosition(a);
    cpb = PointTraits3<B>::getPosition(b);
    return MetricL2::monotoneApproxDistance(cpa, cpb);
  }
};

template <typename B>
struct MetricL2Impl<Ray3, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static double distance(Ray3 const & a, B const & b) { return a.distance(PointTraits3<B>::getPosition(b)); }

  static double monotoneApproxDistance(Ray3 const & a, B const & b)
  { return a.squaredDistance(PointTraits3<B>::getPosition(b)); }

  static double closestPoints(Ray3 const & a, B const & b, Vector3 & cpa, Vector3 & cpb)
  {
    cpb = PointTraits3<B>::getPosition(b);
    cpa = a.closestPoint(cpb);
    return (cpa - cpb).squaredLength();
  }
};

template <typename A>
struct MetricL2Impl<A, Ray3, typename boost::enable_if< IsPoint3<A> >::type>
{
  static double distance(A const & a, Ray3 const & b)
  { return MetricL2Impl<Ray3, A>::distance(b, a); }

  static double monotoneApproxDistance(A const & a, Ray3 const & b)
  { return MetricL2Impl<Ray3, A>::monotoneApproxDistance(b, a); }

  static double closestPoints(A const & a, Ray3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<Ray3, A>::closestPoints(b, a, cpb, cpa); }
};

template <typename B>
struct MetricL2Impl<AxisAlignedBox3, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static double distance(AxisAlignedBox3 const & a, B const & b) { return a.distance(PointTraits3<B>::getPosition(b)); }

  static double monotoneApproxDistance(AxisAlignedBox3 const & a, B const & b)
  { return a.squaredDistance(PointTraits3<B>::getPosition(b)); }

  static double closestPoints(AxisAlignedBox3 const & a, B const & b, Vector3 & cpa, Vector3 & cpb)
  { return monotoneApproxDistance(a, b); /* TODO */ }
};

template <typename A>
struct MetricL2Impl<A, AxisAlignedBox3, typename boost::enable_if< IsPoint3<A> >::type>
{
  static double distance(A const & a, AxisAlignedBox3 const & b)
  { return MetricL2Impl<AxisAlignedBox3, A>::distance(b, a); }

  static double monotoneApproxDistance(A const & a, AxisAlignedBox3 const & b)
  { return MetricL2Impl<AxisAlignedBox3, A>::monotoneApproxDistance(b, a); }

  static double closestPoints(A const & a, AxisAlignedBox3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<AxisAlignedBox3, A>::closestPoints(b, a, cpb, cpa); }
};

template <>
struct MetricL2Impl<AxisAlignedBox3, AxisAlignedBox3>
{
  static double distance(AxisAlignedBox3 const & a, AxisAlignedBox3 const & b) { return a.distance(b); }
  static double monotoneApproxDistance(AxisAlignedBox3 const & a, AxisAlignedBox3 const & b) { return a.squaredDistance(b); }

  static double closestPoints(AxisAlignedBox3 const & a, AxisAlignedBox3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return monotoneApproxDistance(a, b); /* TODO */ }
};

template <typename B>
struct MetricL2Impl<Ball3, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static double distance(Ball3 const & a, B const & b) { return a.distance(PointTraits3<B>::getPosition(b)); }

  static double monotoneApproxDistance(Ball3 const & a, B const & b)
  { return a.squaredDistance(PointTraits3<B>::getPosition(b)); }

  static double closestPoints(Ball3 const & a, B const & b, Vector3 & cpa, Vector3 & cpb)
  { return monotoneApproxDistance(a, b); /* TODO */ }
};

template <typename A>
struct MetricL2Impl<A, Ball3, typename boost::enable_if< IsPoint3<A> >::type>
{
  static double distance(A const & a, Ball3 const & b)
  { return MetricL2Impl<Ball3, A>::distance(b, a); }

  static double monotoneApproxDistance(A const & a, Ball3 const & b)
  { return MetricL2Impl<Ball3, A>::monotoneApproxDistance(b, a); }

  static double closestPoints(A const & a, Ball3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<Ball3, A>::closestPoints(b, a, cpb, cpa); }
};

template <>
struct MetricL2Impl<Ball3, Ball3>
{
  static double distance(Ball3 const & a, Ball3 const & b) { return a.distance(b); }
  static double monotoneApproxDistance(Ball3 const & a, Ball3 const & b) { return a.squaredDistance(b); }

  static double closestPoints(Ball3 const & a, Ball3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return monotoneApproxDistance(a, b); /* TODO */ }
};

template <>
struct MetricL2Impl<Ball3, AxisAlignedBox3>
{
  static double distance(Ball3 const & a, AxisAlignedBox3 const & b) { return a.distance(b); }
  static double monotoneApproxDistance(Ball3 const & a, AxisAlignedBox3 const & b) { return a.squaredDistance(b); }

  static double closestPoints(Ball3 const & a, AxisAlignedBox3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return monotoneApproxDistance(a, b); /* TODO */ }
};

template <>
struct MetricL2Impl<AxisAlignedBox3, Ball3>
{
  static double distance(AxisAlignedBox3 const & a, Ball3 const & b)
  { return MetricL2Impl<Ball3, AxisAlignedBox3>::distance(b, a); }

  static double monotoneApproxDistance(AxisAlignedBox3 const & a, Ball3 const & b)
  { return MetricL2Impl<Ball3, AxisAlignedBox3>::monotoneApproxDistance(b, a); }

  static double closestPoints(AxisAlignedBox3 const & a, Ball3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<Ball3, AxisAlignedBox3>::closestPoints(b, a, cpb, cpa); }
};

template <typename VertexTripleT, typename B>
struct MetricL2Impl<Triangle3<VertexTripleT>, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  typedef Triangle3<VertexTripleT> Triangle;

  static double distance(Triangle const & a, B const & b) { return a.distance(PointTraits3<B>::getPosition(b)); }

  static double monotoneApproxDistance(Triangle const & a, B const & b)
  { return a.squaredDistance(PointTraits3<B>::getPosition(b)); }

  static double closestPoints(Triangle const & a, B const & b, Vector3 & cpa, Vector3 & cpb)
  {
    cpb = PointTraits3<B>::getPosition(b);
    cpa = a.closestPoint(cpb);
    return (cpa - cpb).squaredLength();
  }
};

template <typename A, typename VertexTripleT>
struct MetricL2Impl<A, Triangle3<VertexTripleT>, typename boost::enable_if< IsPoint3<A> >::type>
{
  typedef Triangle3<VertexTripleT> Triangle;

  static double distance(A const & a, Triangle const & b)
  { return MetricL2Impl<Triangle, A>::distance(b, a); }

  static double monotoneApproxDistance(A const & a, Triangle const & b)
  { return MetricL2Impl<Triangle, A>::monotoneApproxDistance(b, a); }

  static double closestPoints(A const & a, Triangle const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<Triangle, A>::closestPoints(b, a, cpb, cpa); }
};

template <typename VertexTripleT1, typename VertexTripleT2>
struct MetricL2Impl< Triangle3<VertexTripleT1>, Triangle3<VertexTripleT2> >
{
  typedef Triangle3<VertexTripleT1> Triangle3_1;
  typedef Triangle3<VertexTripleT2> Triangle3_2;

  static double distance(Triangle3_1 const & a, Triangle3_2 const & b) { return a.distance(b); }
  static double monotoneApproxDistance(Triangle3_1 const & a, Triangle3_2 const & b) { return a.squaredDistance(b); }

  static double closestPoints(Triangle3_1 const & a, Triangle3_2 const & b, Vector3 & cpa, Vector3 & cpb)
  { return a.closestPoints(b, cpa, cpb); }
};

template <typename VertexTripleT>
struct MetricL2Impl< Triangle3<VertexTripleT>, Ball3 >
{
  typedef Triangle3<VertexTripleT> Triangle;

  static double distance(Triangle const & a, Ball3 const & b) { return a.distance(b); }
  static double monotoneApproxDistance(Triangle const & a, Ball3 const & b) { return a.squaredDistance(b); }

  static double closestPoints(Triangle const & a, Ball3 const & b, Vector3 & cpa, Vector3 & cpb)
  { return a.closestPoints(b, cpa, cpb); }
};

template <typename VertexTripleT>
struct MetricL2Impl< Ball3, Triangle3<VertexTripleT> >
{
  typedef Triangle3<VertexTripleT> Triangle;

  static double distance(Ball3 const & a, Triangle const & b)
  { return MetricL2Impl<Triangle, Ball3>::distance(b, a); }

  static double monotoneApproxDistance(Ball3 const & a, Triangle const & b)
  { return MetricL2Impl<Triangle, Ball3>::monotoneApproxDistance(b, a); }

  static double closestPoints(Ball3 const & a, Triangle const & b, Vector3 & cpa, Vector3 & cpb)
  { return MetricL2Impl<Triangle, Ball3>::closestPoints(b, a, cpb, cpa); }
};

/**
 * Helper class for IntersectionTester. Specializations of this class actually test for intersection. This is required because
 * C++ does unexpected things with specialized and overloaded function templates (see
 * http://www.gotw.ca/publications/mill17.htm).
 *
 * @see IntersectionTester
 */
template <typename A, typename B, typename Enable = void>
struct IntersectionTesterImpl
{
  /** Check if two objects intersect. */
  static bool intersects(A const & a, B const & b) { return a.intersects(b); }

}; // class IntersectionTesterImpl

/**
 * %Intersection queries on objects.
 *
 * To add support for intersection queries between custom types, add specializations (full or partial) of the helper class
 * IntersectionTesterImpl. This class is required because C++ does unexpected things with specialized and overloaded function
 * templates (see http://www.gotw.ca/publications/mill17.htm). Note that to commutatively support queries between distinct types
 * A and B, you must specialize both IntersectionTesterImpl<A, B> and IntersectionTesterImpl<B, A>. Do <b>not</b> try
 * specializing IntersectionTester::intersects().
 */
class IntersectionTester
{
  public:
    /**
     * Check if two objects intersect, via the helper class IntersectionTesterImpl. Add specializations of
     * IntersectionTesterImpl as required for specific types of objects. Note that to commutatively support intersection queries
     * on distinct types A and B, you must explicitly specialize both IntersectionTesterImpl<A, B> and
     * IntersectionTesterImpl<B, A>.
     */
    template <typename A, typename B> static bool intersects(A const & a, B const & b)
    { return IntersectionTesterImpl<A, B>::intersects(a, b); }

}; // class IntersectionTester

// Support for pointer types
template <typename A, typename B>
struct IntersectionTesterImpl<A *, B *>
{
  static bool intersects(A const * a, B const * b) { return IntersectionTester::intersects(*a, *b); }
};

template <typename A, typename B>
struct IntersectionTesterImpl<A, B *>
{
  static bool intersects(A const & a, B const * b) { return IntersectionTester::intersects(a, *b); }
};

template <typename A, typename B>
struct IntersectionTesterImpl<A *, B>
{
  static bool intersects(A const * a, B const & b) { return IntersectionTester::intersects(*a, b); }
};

// Default specializations
template <typename B>
struct IntersectionTesterImpl<AxisAlignedBox3, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static bool intersects(AxisAlignedBox3 const & a, B const & b) { return a.intersects(PointTraits3<B>::getPosition(b)); }
};

template <typename A>
struct IntersectionTesterImpl<A, AxisAlignedBox3, typename boost::enable_if< IsPoint3<A> >::type>
{
  static bool intersects(A const & a, AxisAlignedBox3 const & b) { return IntersectionTester::intersects(b, a); }
};

template <typename B>
struct IntersectionTesterImpl<Ball3, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static bool intersects(Ball3 const & a, B const & b) { return a.intersects(PointTraits3<B>::getPosition(b)); }
};

template <typename A>
struct IntersectionTesterImpl<A, Ball3, typename boost::enable_if< IsPoint3<A> >::type>
{
  static bool intersects(A const & a, Ball3 const & b) { return IntersectionTester::intersects(b, a); }
};

template <typename VertexTripleT, typename B>
struct IntersectionTesterImpl<Triangle3<VertexTripleT>, B, typename boost::enable_if< IsPoint3<B> >::type>
{
  static bool intersects(Triangle3<VertexTripleT> const & a, B const & b)
  { return a.intersects(PointTraits3<B>::getPosition(b)); }
};

template <typename A, typename VertexTripleT>
struct IntersectionTesterImpl<A, Triangle3<VertexTripleT>, typename boost::enable_if< IsPoint3<A> >::type>
{
  static bool intersects(A const & a, Triangle3<VertexTripleT> const & b) { return IntersectionTester::intersects(b, a); }
};

/**
 * Helper class for RayIntersectionTester. Specializations of this class actually test for intersection. This is required
 * because C++ does unexpected things with specialized and overloaded function templates (see
 * http://www.gotw.ca/publications/mill17.htm).
 *
 * The default implementation works for objects implementing the RayIntersectable3 interface.
 *
 * @see RayIntersectionTester
 */
template <typename T>
struct RayIntersectionTesterImpl
{
  /**
   * Get the time taken for a ray to intersect an object.
   *
   * @param ray The ray to test for intersection with the object.
   * @param obj The object to test for intersection with the ray.
   * @param max_time Maximum allowable hit time, ignored if negative.
   */
  static Real rayIntersectionTime(Ray3 const & ray, T const & obj, Real max_time = -1)
  { return obj.rayIntersectionTime(ray, max_time); }

  /**
   * Get a description of the intersection point of a ray with an object.
   *
   * @param ray The ray to test for intersection with the object.
   * @param obj The object to test for intersection with the ray.
   * @param max_time Maximum allowable hit time, ignored if negative.
   */
  static RayIntersection3 rayIntersection(Ray3 const & ray, T const & obj, Real max_time = -1)
  { return obj.rayIntersection(ray, max_time); }

}; // struct RayIntersectionTesterImpl

/**
 * Ray intersection queries on objects.
 *
 * To add support for intersection queries between custom types, add specializations (full or partial) of the helper class
 * RayIntersectionTesterImpl. This class is required because C++ does unexpected things with specialized and overloaded function
 * templates (see http://www.gotw.ca/publications/mill17.htm). Do <b>not</b> try specializing
 * RayIntersectionTester::rayIntersectionTime() or RayIntersectionTester::rayIntersection().
 */
class RayIntersectionTester
{
  public:
    /**
     * Get the time taken for a ray to intersect an object, via the helper class RayIntersectionTesterImpl. Add specializations
     * of RayIntersectionTesterImpl as required.
     *
     * @param ray The ray to test for intersection with the object.
     * @param obj The object to test for intersection with the ray.
     * @param max_time Maximum allowable hit time, ignored if negative.
     */
    template <typename T> static Real rayIntersectionTime(Ray3 const & ray, T const & obj, Real max_time = -1)
    {
      return RayIntersectionTesterImpl<T>::rayIntersectionTime(ray, obj, max_time);
    }

    /**
     * Get a description of the intersection point of a ray with an object, via the helper class RayIntersectionTesterImpl. Add
     * specializations of RayIntersectionTesterImpl as required.
     *
     * @param ray The ray to test for intersection with the object.
     * @param obj The object to test for intersection with the ray.
     * @param max_time Maximum allowable hit time, ignored if negative.
     */
    template <typename T> static RayIntersection3 rayIntersection(Ray3 const & ray, T const & obj, Real max_time = -1)
    {
      return RayIntersectionTesterImpl<T>::rayIntersection(ray, obj, max_time);
    }

}; // class RayIntersectionTester

// Support for pointer types
template <typename T>
struct RayIntersectionTesterImpl<T *>
{
  static Real rayIntersectionTime(Ray3 const & ray, T const * obj, Real max_time = -1)
  { return RayIntersectionTester::rayIntersectionTime(ray, *obj, max_time); }

  static RayIntersection3 rayIntersection(Ray3 const & ray, T const * obj, Real max_time = -1)
  { return RayIntersectionTester::rayIntersection(ray, *obj, max_time); }
};

/**
 * A sorted array of a given maximum size, ordered in ascending order according to a comparator. If the array is full and a new
 * one is added, the last element is dropped.
 *
 * To get some extra speed when T has a trivial (bit-copy) assignment operator, make sure that
 * <tt>boost::has_trivial_assign</tt> is true for T.
 *
 * The implementation always allocates enough space to store N instances of T.
 */
template < typename T, typename Compare = std::less<T> >
class BoundedSortedArray
{
  private:
    Compare compare;
    int capacity, num_elems;
    T * values;

  public:
    /**
     * Constructor. Allocates memory for \a capacity_ elements.
     *
     * @param capacity_ The maximum number of elements the array can hold. Must be a positive (non-zero) integer.
     * @param compare_ The comparator that evaluates the "less-than" operator on objects of type T.
     */
    BoundedSortedArray(int capacity_, Compare compare_ = Compare()) : compare(compare_), capacity(capacity_), num_elems(0)
    {
      THEA_ASSERT(capacity > 0, "BoundedSortedArray: Capacity must be positive");
      values = new T[capacity];
    }

    /** Copy constructor. */
    BoundedSortedArray(BoundedSortedArray const & src)
    : compare(src.compare), capacity(src.capacity), num_elems(src.num_elems), values(new T[src.capacity])
    {
      if (src.num_elems > 0)
        fastCopy(src.values, src.values + src.num_elems, values);
    }

    /** Assignment operator. */
    BoundedSortedArray & operator=(BoundedSortedArray const & src)
    {
      compare = src.compare;
      num_elems = src.num_elems;

      if (capacity != src.capacity)
      {
        delete [] values;
        values = new T[src.capacity];
        capacity = src.capacity;
      }

      if (src.num_elems > 0)
        fastCopy(src.values, src.values + src.num_elems, values);

      return *this;
    }

    /** Destructor. */
    ~BoundedSortedArray() { delete [] values; }

    /** Get the maximum number of elements the array can hold. */
    int getCapacity() const { return capacity; }

    /** Get the number of elements in the array. */
    int size() const { return num_elems; }

    /** Check if the array is empty or not. */
    bool isEmpty() const { return num_elems <= 0; }

    /** Get the first element in the sorted sequence. */
    T const & first() const
    {
      THEA_ASSERT(num_elems > 0, "BoundedSortedArray: Can't get first element of empty array");
      return values[0];
    }

    /** Get the last element in the sorted sequence. */
    T const & last() const
    {
      THEA_ASSERT(num_elems > 0, "BoundedSortedArray: Can't get last element of empty array");
      return values[num_elems - 1];
    }

    /** Get the element at a given position in the sorted sequence. */
    T const & operator[](int i) const
    {
      THEA_ASSERT(i >= 0 && i < capacity, "BoundedSortedArray: Index out of bounds");
      return values[i];
    }

    /**
     * Get the index of a given value, or negative if it is not present in the array. If the value occurs multiple times, the
     * index of any one occurrence is returned.
     */
    int find(T const & t) const
    {
      int lb = lowerBound(t);
      return (lb < num_elems && !compare(t, values[lb])) ? lb : -1;
    }

    /**
     * Get the index of the first element strictly greater than \a t, or return the capacity of the array if no such element is
     * present.
     */
    int upperBound(T const & t) const
    {
      int first = 0, mid, step;
      int count = num_elems;
      while (count > 0)
      {
        step = count >> 1;
        mid = first + step;
        if (!compare(t, values[mid]))
        {
          first = mid + 1;
          count -= (step + 1);
        }
        else
          count = step;
      }

      return first;
    }

    /**
     * Get the index of the first element equal to or greater than \a t, or return the capacity of the array if no such element
     * is present.
     */
    int lowerBound(T const & t) const
    {
      int first = 0, mid, step;
      int count = num_elems;
      while (count > 0)
      {
        step = count >> 1;
        mid = first + step;
        if (compare(values[mid], t))
        {
          first = mid + 1;
          count -= (step + 1);
        }
        else
          count = step;
      }

      return first;
    }

    /**
     * Check if a value can be inserted in the array. This requires that either the array has fewer elements than its capacity,
     * or the value is "less than" the last element in the array.
     */
    bool isInsertable(T const & t) const
    {
      return num_elems < capacity || compare(t, last());
    }

    /**
     * Insert a value into the array.
     *
     * @return The index of the newly inserted element, or negative if the value could not be inserted.
     */
    int insert(T const & t)
    {
      if (num_elems <= 0)
      {
        values[0] = t;
        ++num_elems;
        return 0;
      }
      else if (isInsertable(t))
      {
        int ub = upperBound(t);
        T * end = values + (num_elems < capacity ? num_elems : capacity - 1);
        fastCopyBackward(values + ub, end, end + 1);
        values[ub] = t;
        if (num_elems < capacity) ++num_elems;
        return ub;
      }

      return -1;
    }

    /** Remove the element at the given position from the array. */
    void erase(int i)
    {
      if (i >= 0 && i < num_elems)
      {
        fastCopy(values + i + 1, values + num_elems, values + i);
        --num_elems;
      }
    }

    /** Remove (one occurrence of) the given value from the array, if it is present. */
    void erase(T const & t)
    {
      erase(find(t));
    }

    /** Remove all elements from the array. */
    void clear()
    {
      num_elems = 0;
    }

}; // class BoundedSortedArray

} // namespace Thea

#endif
