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

#ifndef __Thea_KDTree3_hpp__
#define __Thea_KDTree3_hpp__

#include "Util.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace Thea {

/**
 * Get the estimated depth of a kd-tree with a given number of elements, assuming an upper bound on the number of elements in
 * each leaf, and the average split ratio at each node.
 *
 * @param num_elems The number of elements in the tree.
 * @param max_elems_in_leaf The maximum number of elements allowed in each leaf.
 * @param split_ratio The average splitting ratio at a node, expressed as the fraction of elements in the larger child of the
 *   node (0.5 is a perfectly fair split). Must be in the range (0, 1).
 */
inline int
kdtreeDepth(long num_elems, int max_elems_in_leaf, Real split_ratio = 0.5)
{
  THEA_ASSERT(num_elems >= 0, "Math: Can't compute kd-tree depth for negative number of elements");
  THEA_ASSERT(max_elems_in_leaf > 0, "Math: Can't compute kd-tree depth for non-positive number of elements at leaf");
  THEA_ASSERT(split_ratio > 0 && split_ratio < 1, "Math: KD-tree split ratio must be in range (0, 1)");

  if (num_elems <= 0) return 0;

  double log_inv_split_ratio = -std::log(split_ratio);
  int est_depth = (int)std::ceil(std::log(num_elems / (double)max_elems_in_leaf) / log_inv_split_ratio);

  return est_depth < 0 ? 0 : est_depth;  // check shouldn't be necessary but do it just in case
}

/** A pair of points in 3-space. */
typedef std::pair<Vector3, Vector3> PointPair3;

/** A pair of indices. */
typedef std::pair<long, long> IndexPair;

class ProximityQueryStructure3
{
  public:
    /**
     * A return value of a k-nearest-neighbors query, specified by a monotone approximation to (for L2, square of) the distance
     * to the neighbor, and the index of the neighbor.
     */
    class Neighbor
    {
      public:
        Neighbor() {}
        Neighbor(double mon_approx_dist_, long index_) : mon_approx_dist(mon_approx_dist_), index(index_) {}

        double getMonotoneApproxDistance() const { return mon_approx_dist; }
        void setMonotoneApproxDistance(double mon_approx_dist_) { mon_approx_dist = mon_approx_dist_; }

        long getIndex() const { return index; }
        void setIndex(long index_) { index = index_; }

        // Use the index to break ties
        bool operator<(Neighbor const & other) const
        { return mon_approx_dist < other.mon_approx_dist || (mon_approx_dist == other.mon_approx_dist && index < other.index); }

      private:
        double mon_approx_dist;
        long index;
    };

    /** Get a bounding box for the structure. */
    AxisAlignedBox3 const & getBounds() const;

    /** Get the minimum distance between this structure and a query object. */
    template <typename MetricT, typename QueryT> double distance(QueryT const & query, double dist_bound = -1) const;

    /**
     * Get the closest element in this structure to a query object, within a specified distance bound.
     *
     * @param query Query object.
     * @param dist_bound Upper bound on the distance between any pair of points considered. Ignored if negative.
     * @param dist The distance to the query object is placed here. Ignored if null.
     * @param closest_point The coordinates of the closest point are placed here. Ignored if null.
     *
     * @return A non-negative handle to the closest element, if one was found, else a negative number.
     */
    template <typename MetricT, typename QueryT>
    long closestElement(QueryT const & query, double dist_bound = -1, double * dist = NULL, Vector3 * closest_point = NULL)
         const;

    /**
     * Get the closest pair of elements between this structure and another structure, whose separation is less than a specified
     * upper bound.
     *
     * @param query Query object.
     * @param dist_bound Upper bound on the distance between any pair of points considered. Ignored if negative.
     * @param dist The distance between the closest pair of points is placed here. Ignored if null.
     * @param closest_points The coordinates of the closest pair of points are placed here. Ignored if null.
     *
     * @return Non-negative handles to the closest pair of elements in their respective objects, if such a pair was found. Else
     *   returns a pair of negative numbers.
     */
    template <typename MetricT, typename QueryT>
    IndexPair closestPair(QueryT const & query, double dist_bound = -1, double * dist = NULL,
                          PointPair3 * closest_points = NULL) const;

    /**
     * Get the k elements closest to a query object.
     *
     * @param query Query object.
     * @param k_closest_elems The k (or fewer) nearest neighbors are placed here.
     * @param dist_bound Upper bound on the distance between any pair of points considered. Ignored if negative.
     * @param clear_set If true (default), this function discards prior data in \a k_closest_elems. This is chiefly for internal
     *   use and the default value of true should normally be left as is.
     */
    template <typename MetricT, typename QueryT>
    void kClosestElements(QueryT const & query, BoundedSortedArray<Neighbor> & k_closest_elems, double dist_bound = -1,
                          bool clear_set = true) const;

}; // class ProximityQueryStructure3

/**
 * Interface for a structure that supports range queries in 3-space. None of the functions are virtual, this just defines a
 * concept subclasses must implement.
 */
template <typename T>
class RangeQueryStructure3
{
  public:
    /** Get all objects intersecting a range. */
    template <typename IntersectionTesterT, typename RangeT>
    void rangeQuery(RangeT const & range, std::vector<T> & result) const;

    /** Get the indices of all objects intersecting a range. */
    template <typename IntersectionTesterT, typename RangeT>
    void rangeQueryIndices(RangeT const & range, std::vector<long> & result) const;

    /**
     * Apply a functor to all objects in a range, until the functor returns true. The functor should provide the member function
     * (or be a function pointer with the equivalent signature)
     * \code
     * bool operator()(long index, T & t)
     * \endcode
     * and will be passed the index of each object contained in the range as well as a handle to the object itself. If the
     * functor returns true on any object, the search will terminate immediately (this is useful for searching for a particular
     * object).
     *
     * @return True if the functor evaluated to true on any object in the range (and hence stopped immediately after processing
     *   this object), else false.
     */
    template <typename IntersectionTesterT, typename RangeT, typename FunctorT>
    bool processRangeUntil(RangeT const & range, FunctorT & functor);

}; // class RangeQueryStructure3

/**
 * A description of the intersection point of a ray with a structure. Specifies the hit time, the normal at the intersection
 * point, and the index of the intersected element.
 */
class RayStructureIntersection3 : public RayIntersection3
{
  private:
    typedef RayIntersection3 BaseT;

    long element_index;

  public:
    /** Constructor. */
    RayStructureIntersection3(Real time_ = -1, Vector3 const * normal_ = NULL, long element_index_ = -1)
    : BaseT(time_, normal_), element_index(element_index_)
    {}

    /** Constructor. */
    RayStructureIntersection3(RayIntersection3 const & isec, long element_index_ = -1)
    : BaseT(isec), element_index(element_index_)
    {}

    /** Get the index of the intersected element. */
    long getElementIndex() const { return element_index; }

    /** Set the index of the intersected element. */
    void setElementIndex(long element_index_) { element_index = element_index_; }

}; // class RayStructureIntersection3

/**
 * Interface for a structure that supports ray intersection queries in 3-space. None of the functions are virtual, this just
 * defines a concept subclasses must implement.
 */
class RayQueryStructure3
{
  public:
    /** Check if a ray intersects the structure in the forward direction. */
    template <typename RayIntersectionTesterT> bool rayIntersects(Ray3 const & ray) const;

    /**
     * Get the time taken for a ray to intersect the structure, or a negative value if there was no intersection in the forward
     * direction.
     */
    template <typename RayIntersectionTesterT> Real rayIntersectionTime(Ray3 const & ray) const;

    /**
     * Get the intersection of a ray with the structure, including the hit time, the normal at the intersection point, and the
     * index of the intersected element. A negative time is returned if there was no intersection in the forward direction. A
     * zero normal and a negative index are returned if those quantities are not known.
     */
    template <typename RayIntersectionTesterT> RayStructureIntersection3 rayStructureIntersection(Ray3 const & ray) const;

}; // class RayQueryStructure3

/**
 * A kd-tree for a set of objects in 3-space. BoundedObjectTraits3 must be specialized for the type T.
 *
 * @todo Associate elements that intersect splitting planes with the parent node, instead of with both child nodes? This has
 *   pros (less wasted space) and cons (less pruning for NN queries).
 */
template <typename T>
class KDTree3
: public RangeQueryStructure3<T>,
  public ProximityQueryStructure3,
  public RayQueryStructure3,
  private Noncopyable
{
  public:
    typedef size_t ElementIndex;  ///< Index of an element in the kd-tree.

  private:
    typedef RangeQueryStructure3<T>   RangeQueryBaseT;
    typedef ProximityQueryStructure3  ProximityQueryBaseT;
    typedef RayQueryStructure3        RayQueryBaseT;

    /**
     * A memory pool for fast allocation of arrays. A pointer to memory allocated from the pool remains valid, and the pool
     * never reduces in size, until the pool is cleared, reinitialized, or destroyed.
     */
    template <typename U> class MemoryPool : private Noncopyable
    {
      private:
        /** A single buffer in the memory pool. */
        class Buffer : private Noncopyable
        {
          public:
            /** Constructor. */
            Buffer(size_t capacity_ = 0) : data(NULL), capacity(capacity_), current_end(0)
            {
              THEA_ASSERT(capacity_ > 0, "KDTree3: Memory pool buffer capacity must be positive");
              if (capacity > 0)
              {
                data = new U[capacity];
                // THEA_CONSOLE << "Allocated data block " << data << " for buffer " << this;
              }
            }

            /** Destructor. */
            ~Buffer()
            {
              // THEA_CONSOLE << "Deleting data " << data << " of buffer " << this;
              delete [] data;
            }

            /** Clears the buffer without deallocating buffer memory. */
            void reset()
            {
              current_end = 0;
            }

            /**
             * Allocate a block of elements and return a pointer to the first allocated element, or null if the allocation
             * exceeded buffer capacity.
             */
            U * alloc(size_t num_elems)
            {
              // THEA_CONSOLE << "KDTree3: Allocating " << num_elems << " elements from buffer of capacity " << capacity;

              if (current_end + num_elems > capacity)
                return NULL;

              U * ret = &data[current_end];
              current_end += num_elems;
              return ret;
            }

            /** Deallocate a block of elements and return the number of elements successfully deallocated. */
#if defined(_MSC_VER) && defined(_DEBUG)
            size_t _free(size_t num_elems)
#else
            size_t free(size_t num_elems)
#endif
            {
              if (num_elems > current_end)
              {
                size_t num_freed = current_end;
                current_end = 0;
                return num_freed;
              }

              current_end -= num_elems;
              return num_elems;
            }

            /** Element access. */
            U const & operator[](size_t i) const { return data[i]; }

            /** Element access. */
            U & operator[](size_t i) { return data[i]; }

          private:
            U * data;
            size_t capacity;
            size_t current_end;

        }; // Buffer

        std::vector<Buffer *> buffers;
        size_t buffer_capacity;
        int current_buffer;

        /** Get the current buffer. */
        Buffer & getCurrentBuffer() { return *buffers[(size_t)current_buffer]; }

        /** Get the next available buffer, creating it if necessary and making it the current one. */
        Buffer & getNextBuffer()
        {
          int next_buffer = current_buffer < 0 ? 0 : current_buffer + 1;
          if ((size_t)next_buffer >= buffers.size())
          {
            buffers.push_back(new Buffer(buffer_capacity));

            // THEA_CONSOLE << "KDTree3: Added buffer to memory pool " << this << ", current_buffer = " << current_buffer
            //              << ", next_buffer = " << next_buffer;
          }

          current_buffer = next_buffer;
          return *buffers[(size_t)current_buffer];
        }

      public:
        /** Constructor. */
        MemoryPool() : buffer_capacity(0), current_buffer(-1)
        {
          // THEA_CONSOLE << "KDTree3: Creating memory pool " << this;
        }

        /** Destructor. */
        ~MemoryPool()
        {
          clear(true);
          // THEA_CONSOLE << "KDTree3: Destroyed memory pool " << this;
        }

        /** Initialize the memory pool to hold buffers of a given capacity. Previous data in the pool is deallocated. */
        void init(size_t buffer_capacity_)
        {
          // THEA_CONSOLE << "KDTree3: Initializing memory pool " << this << " with buffer capacity " << buffer_capacity_
          //              << " elements";

          clear(true);

          buffer_capacity = buffer_capacity_;
        }

        /** Get the maximum number of elements of type T that a single buffer can hold. */
        size_t getBufferCapacity() const
        {
          return buffer_capacity;
        }

        /** Reset the memory pool, optionally deallocating and removing all buffers. */
        void clear(bool deallocate_all_memory = true)
        {
          // THEA_CONSOLE << "KDTree3: Clearing memory pool " << this;

          if (deallocate_all_memory)
          {
            for (size_t i = 0; i < buffers.size(); ++i)
              delete buffers[i];

            buffers.clear();
          }
          else
          {
            for (size_t i = 0; i < buffers.size(); ++i)
              buffers[i]->reset();
          }

          current_buffer = -1;
        }

        /** Allocate a block of elements from the pool and return a pointer to the first allocated element. */
        U * alloc(size_t num_elems)
        {
          THEA_ASSERT(num_elems <= buffer_capacity, "KDTree3: A single memory pool allocation cannot exceed buffer capacity");

          if (current_buffer >= 0)
          {
            U * ret = getCurrentBuffer().alloc(num_elems);
            if (!ret)
              return getNextBuffer().alloc(num_elems);
            else
              return ret;
          }

          return getNextBuffer().alloc(num_elems);
        }

        /** Deallocate a block of elements from the pool. */
#if defined(_MSC_VER) && defined(_DEBUG)
        void _free(size_t num_elems)
#else
        void free(size_t num_elems)
#endif
        {
          long n = (long)num_elems;
          while (n > 0 && current_buffer >= 0)
          {
#if defined(_MSC_VER) && defined(_DEBUG)
        size_t num_freed = getCurrentBuffer()._free(num_elems);
#else
        size_t num_freed = getCurrentBuffer().free(num_elems);
#endif                
            n -= (long)num_freed;

            if (n > 0)
              current_buffer--;
          }
        }

    }; // class MemoryPool

    /** A functor to add results of a range query to an array. */
    class RangeQueryFunctor
    {
      public:
        RangeQueryFunctor(std::vector<T> & result_) : result(result_) {}
        bool operator()(long index, T & t) { result.push_back(t); return false; }

      private:
        std::vector<T> & result;
    };

    /** A functor to add the indices of results of a range query to an array. */
    class RangeQueryIndicesFunctor
    {
      public:
        RangeQueryIndicesFunctor(std::vector<long> & result_) : result(result_) {}
        bool operator()(long index, T & t) { result.push_back(index); return false; }

      private:
        std::vector<long> & result;
    };

  public:
    typedef T                        Element;              ///< The type of elements in the kd-tree.
    typedef T                        value_type;           ///< The type of elements in the kd-tree (STL convention).
    typedef BoundedObjectTraits3<T>  BoundedObjectTraits;  ///< Gives bounding volumes for elements.

    /** A node of the kd-tree. Only immutable objects of this class should be exposed by the external kd-tree interface. */
    class Node
    {
      private:
        int depth;
        AxisAlignedBox3 bounds;
        size_t num_elems;
        ElementIndex * elems;
        Node * lo;
        Node * hi;

        friend class KDTree3;

        void init(int depth_)
        {
          depth = depth_;
          bounds = AxisAlignedBox3::zero();
          num_elems = 0;
          elems = NULL;
          lo = hi = NULL;
        }

      public:
        /** Iterator over immutable element indices. Dereferences to an array index. */
        typedef ElementIndex const * ElementIndexConstIterator;

        /** Constructor. */
        Node(int depth_ = 0) : depth(depth_), bounds(AxisAlignedBox3::zero()), lo(NULL), hi(NULL) {}

        /** Get the depth of the node in the tree (the root is at depth 0). */
        int getDepth() const { return depth; }

        /** Get the bounding box of the node. */
        AxisAlignedBox3 const & getBounds() const { return bounds; }

        /**
         * Get the number of element indices stored at this node. This is <b>not</b> the number of elements within the node's
         * bounding box: in memory-saving mode, indices of all such elements are only held at the leaves of the subtree rooted
         * at this node.
         */
        long numElementIndices() const { return (long)num_elems; }

        /** Get an iterator to the first element index stored at the node. */
        ElementIndexConstIterator elementIndicesBegin() const { return elems; }

        /** Get an iterator to one past the last element index stored at the node. */
        ElementIndexConstIterator elementIndicesEnd() const { return elems + num_elems; }

        /** Get the child corresponding to the lower half of the range. */
        Node const * getLowChild() const { return lo; }

        /** Get the child containing the upper half of the range. */
        Node const * getHighChild() const { return hi; }

        /** Check if the node is a leaf (both children are null) are not. */
        bool isLeaf() const { return !(lo || hi); }

    }; // Node

  private:
    typedef MemoryPool<Node> NodePool;  ///< A pool for quickly allocating kd-tree nodes.
    typedef MemoryPool<ElementIndex> IndexPool;  ///< A pool for quickly allocating element indices.
    typedef std::vector<T> ElementArray;  ///< An array of elements.

  public:
    /** Default constructor. */
    KDTree3() : root(NULL), num_elems(0), num_nodes(0), max_depth(0), max_elems_in_leaf(0) {}

    /**
     * Construct from a list of elements. InputIterator must dereference to type T.
     *
     * @param begin Points to the first element to be added.
     * @param end Points to one position beyond the last element to be added.
     * @param max_depth_ Maximum depth of the tree. The root is at depth zero. Use a negative argument to auto-select a suitable
     *   value.
     * @param max_elems_in_leaf_ Maximum number of elements in a leaf (unless the depth exceeds the maximum). Use a negative
     *   argument to auto-select a suitable value.
     * @param save_memory If true, element references at inner nodes of the tree are deleted to save memory. This could slow
     *   down range searches since every positive result will only be obtained at the leaves.
     */
    template <typename InputIterator>
    KDTree3(InputIterator begin, InputIterator end, int max_depth_ = -1, int max_elems_in_leaf_ = -1, bool save_memory = false)
    : root(NULL), num_elems(0), num_nodes(0), max_depth(0), max_elems_in_leaf(0), valid_bounds(true),
      bounds(AxisAlignedBox3::zero())
    {
      init(begin, end, max_elems_in_leaf_, max_depth_, save_memory, false /* no previous data to deallocate */);
    }

    /**
     * Construct from a list of elements. InputIterator must dereference to type T. Any previous data is discarded. If any
     * filters are active at this time, only those input elements that pass the filters will be retained in the tree.
     *
     * @param begin Points to the first element to be added.
     * @param end Points to one position beyond the last element to be added.
     * @param max_depth_ Maximum depth of the tree. The root is at depth zero. Use a negative argument to auto-select a suitable
     *   value.
     * @param max_elems_in_leaf_ Maximum number of elements in a leaf (unless the depth exceeds the maximum). Use a negative
     *   argument to auto-select a suitable value.
     * @param save_memory If true, element references at inner nodes of the tree are deleted to save memory. This could slow
     *   down range searches since every positive result will only be obtained at the leaves.
     * @param deallocate_previous_memory If true, all previous data held in internal memory pools is explicitly deallocated.
     *   Else, all such space is reused and overwritten when possible. If \a save_memory is true, or some filters are active,
     *   this flag may not be quite as effective since it's more likely that some space will be allocated/deallocated. Note that
     *   if this flag is set to false, the space used internally by the kd-tree will not decrease except in some special
     *   implementation-specific cases.
     */
    template <typename InputIterator>
    void init(InputIterator begin, InputIterator end, int max_depth_ = -1, int max_elems_in_leaf_ = -1,
              bool save_memory = false, bool deallocate_previous_memory = true)
    {
      clear(deallocate_previous_memory);

      if (deallocate_previous_memory)
      {
        for (InputIterator ii = begin; ii != end; ++ii, ++num_elems)
          if (elementPassesFilters(*ii))
            elems.push_back(*ii);
      }
      else
      {
        size_t max_new_elems = (size_t)std::distance(begin, end);
        bool resized = false;
        if (max_new_elems > elems.size())
        {
          if (filters.empty())
            elems.resize((size_t)std::ceil(1.2 * max_new_elems));  // add a little more space to avoid future reallocs
          else
            elems.clear();  // we don't know how many elements will pass the filter

          resized = true;
        }

        if (filters.empty())
        {
          std::copy(begin, end, elems.begin());
          num_elems = max_new_elems;
        }
        else
        {
          if (resized)
          {
            for (InputIterator ii = begin; ii != end; ++ii)
              if (elementPassesFilters(*ii))
              {
                elems.push_back(*ii);
                ++num_elems;
              }
          }
          else
          {
            typename ElementArray::iterator ei = elems.begin();
            for (InputIterator ii = begin; ii != end; ++ii)
              if (elementPassesFilters(*ii))
              {
                *(ei++) = *ii;
                ++num_elems;
              }
          }
        }
      }

      if (num_elems <= 0)
        return;

      static int const DEFAULT_MAX_ELEMS_IN_LEAF  =  10;
      max_elems_in_leaf = max_elems_in_leaf_ < 0 ? DEFAULT_MAX_ELEMS_IN_LEAF : max_elems_in_leaf_;

      // Assume the average fraction of elements held by the larger node at each split is 0.6 (0.5 being the best, denoting a
      // split down the middle)
      static double const BEST_SPLIT = 0.6;
      int est_depth = kdtreeDepth(num_elems, max_elems_in_leaf, BEST_SPLIT);
      max_depth = max_depth_;
      if (max_depth < 0)
        max_depth = est_depth;
      else if (max_depth < est_depth)
        est_depth = max_depth;

      // THEA_CONSOLE << "KDTree3: max_depth = " << max_depth << ", est_depth = " << est_depth;

      // At best, each index is stored at most once at each level. The number increases when an element straddles a splitting
      // boundary, so multiply by 1.5 for a safety margin. With this scheme, we should have about one buffer per level. The
      // final factor of two is because before index compaction, the two children of a node use twice as much space to store
      // their indices as does their parent.
      size_t index_buffer_capacity = (size_t)std::ceil(2 * 1.5 * num_elems);
      if (!save_memory)
        index_buffer_capacity *= (size_t)(1 + est_depth);  // reserve space for all levels at once

      if (deallocate_previous_memory || index_buffer_capacity > 1.3 * index_pool.getBufferCapacity())
      {
        // THEA_CONSOLE << "KDTree3: Resizing index pool: old buffer capacity = " << index_pool.getBufferCapacity()
        //              << ", new buffer capacity = " << index_buffer_capacity;
        index_pool.init(index_buffer_capacity);
      }

      // Assume a complete, balanced binary tree upto the estimated depth to guess the number of leaf nodes, and multiply by 1.5
      // to accommodate elements straddling splitting boundaries.
      size_t node_buffer_capacity = (size_t)std::ceil(1.5 * (1 << est_depth));
      if (deallocate_previous_memory || node_buffer_capacity > 1.3 * node_pool.getBufferCapacity())
      {
        // THEA_CONSOLE << "KDTree3: Resizing node pool: old buffer capacity = " << node_pool.getBufferCapacity()
        //              << ", new buffer capacity = " << node_buffer_capacity;
        node_pool.init(node_buffer_capacity);
      }

      // Create the root node
      root = node_pool.alloc(1);
      root->init(0);
      num_nodes = 1;

      // THEA_CONSOLE << "Allocated root " << root << " from mempool " << &node_pool;

      root->num_elems = num_elems;
      root->elems = index_pool.alloc(root->num_elems);

      AxisAlignedBox3 elem_bounds;
      bool first = true;
      for (size_t i = 0; i < (size_t)num_elems; ++i)
      {
        root->elems[i] = i;

        if (first)
        {
          BoundedObjectTraits::getBounds(elems[i], root->bounds);
          first = false;
        }
        else
        {
          BoundedObjectTraits::getBounds(elems[i], elem_bounds);
          root->bounds.merge(elem_bounds);
        }
      }

      // Expand the bounding box slightly to handle numerical error
      root->bounds.scaleCentered(BOUNDS_EXPANSION_FACTOR);

      if (save_memory)
      {
        // Estimate the maximum number of indices that will need to be held in the scratch pool at any time during depth-first
        // traversal with earliest-possible deallocation. This is
        //
        //      #elements * (sum of series 1 + 2 * (1 + BEST_SPLIT + BEST_SPLIT^2 + ... + BEST_SPLIT^(est_depth - 1)))
        //  <=  #elements * (1 + 2 / (1 - BEST_SPLIT))
        //
        // The estimate is multiplied by 1.5 as a safety margin for elements straddling splitting boundaries.
        //
        size_t est_max_path_indices = (size_t)std::ceil(1.5 * num_elems * (1 + 2 / (1 - BEST_SPLIT)));
        // THEA_CONSOLE << "KDTree3: Estimated maximum number of indices on a single path = " << est_max_path_indices;

        // Create a temporary pool for scratch storage
        IndexPool tmp_index_pool;
        tmp_index_pool.init(est_max_path_indices);

        createTree(root, true, &tmp_index_pool, &index_pool);
      }
      else
        createTree(root, false, &index_pool, NULL);

      invalidateBounds();
    }

    /** Destructor. */
    ~KDTree3() { clear(true); }

    /**
     * Clear the tree. If \a deallocate_all_memory is false, memory allocated in pools is held to be reused if possible by the
     * next init() operation.
     */
    void clear(bool deallocate_all_memory = true)
    {
      num_elems = 0;
      if (deallocate_all_memory)
        elems.clear();

      node_pool.clear(deallocate_all_memory);
      index_pool.clear(deallocate_all_memory);

      root = NULL;

      invalidateBounds();
    }

    /** Get the number of elements in the tree. The elements themselves can be obtained with getElements(). */
    long numElements() const { return num_elems; }

    /** Get a pointer to an array of the elements in the tree. The number of elements can be obtained with numElements(). */
    T const * getElements() const { return &elems[0]; }

    /**
     * Get the node corresponding to the root of the kd-tree. This function is provided so that users can implement their own
     * tree traversal procedures without being restricted by the interface of RangeQueryStructure3.
     *
     * This function cannot be used to change the structure of the tree, or any value in it (unless <code>const_cast</code> is
     * used, which is not recommended).
     *
     * @note An empty tree has a null root.
     */
    Node const * getRoot() const { return root; }

    /** Get the number of nodes in the tree. */
    long numNodes() const { return num_nodes; }

    AxisAlignedBox3 const & getBounds() const
    {
      updateBounds();
      return bounds;
    }

    /**
     * Push an element filter onto the filter stack. Elements in the tree that are not passed by all filters currently on the
     * stack are ignored for all operations, including init().
     *
     * The filter must persist until it is popped off. Must be matched with popFilter().
     *
     * @see popFilter()
     */
    void pushFilter(Filter<T> * filter)
    {
      filters.push_back(filter);
    }

    /**
     * Pops the last pushed element filter off the filter stack. Must be matched with a preceding pushFilter().
     *
     * @see pushFilter()
     */
    void popFilter()
    {
      filters.pop_back();
    }

    template <typename MetricT, typename QueryT> double distance(QueryT const & query, double dist_bound = -1) const
    {
      double result = -1;
      if (closestElement<MetricT>(query, dist_bound, &result) >= 0)
        return result;
      else
        return -1;
    }

    template <typename MetricT, typename QueryT>
    long closestElement(QueryT const & query, double dist_bound = -1, double * dist = NULL, Vector3 * closest_point = NULL)
                        const
    {
      PointPair3 cpp;
      PointPair3 * cpp_ptr = closest_point ? &cpp : NULL;

      IndexPair pair = closestPair<MetricT>(query, dist_bound, dist, cpp_ptr);

      if (closest_point && pair.first >= 0 && pair.second >= 0)
        *closest_point = cpp.first;

      return pair.first;
    }

    // BoundedObjectTraits3<QueryT> must be defined.
    template <typename MetricT, typename QueryT>
    IndexPair closestPair(QueryT const & query, double dist_bound = -1, double * dist = NULL,
                          PointPair3 * closest_points = NULL) const
    {
      if (!root) return IndexPair(-1, -1);

      AxisAlignedBox3 query_bounds;
      BoundedObjectTraits3<QueryT>::getBounds(query, query_bounds);

      // Early pruning if the entire structure is too far away from the query
      double mon_approx_dist_bound = (dist_bound >= 0 ? MetricT::computeMonotoneApprox(dist_bound) : -1);
      if (mon_approx_dist_bound >= 0)
      {
        double lower_bound = MetricT::monotoneApproxDistance(getBoundsWorldSpace(*root), query_bounds);
        if (lower_bound > mon_approx_dist_bound)
          return IndexPair(-1, -1);
      }

      double mon_approx_dist = mon_approx_dist_bound;
      IndexPair pair = closestPair<MetricT>(root, query, query_bounds, mon_approx_dist, closest_points);
      if (dist && pair.first >= 0 && pair.second >= 0)
        *dist = MetricT::invertMonotoneApprox(mon_approx_dist);

      return pair;
    }

    template <typename MetricT, typename QueryT>
    void kClosestElements(QueryT const & query, BoundedSortedArray<Neighbor> & k_closest_elems, double dist_bound = -1,
                          bool clear_set = true) const
    {
      if (clear_set) k_closest_elems.clear();

      if (!root) return;

      AxisAlignedBox3 query_bounds;
      BoundedObjectTraits3<QueryT>::getBounds(query, query_bounds);

      // Early pruning if the entire structure is too far away from the query
      double mon_approx_dist_bound = (dist_bound >= 0 ? MetricT::computeMonotoneApprox(dist_bound) : -1);
      if (mon_approx_dist_bound >= 0)
      {
        double lower_bound = MetricT::monotoneApproxDistance(getBoundsWorldSpace(*root), query_bounds);
        if (lower_bound > mon_approx_dist_bound)
          return;

        if (!k_closest_elems.isInsertable(Neighbor(lower_bound, 0)))
          return;
      }

      kClosestElements<MetricT>(root, query, query_bounds, k_closest_elems, dist_bound);
    }

    template <typename RangeT>
    void rangeQuery(RangeT const & range, std::vector<T> & result) const
    {
      if (root)
      {
        RangeQueryFunctor functor(result);
        const_cast<KDTree3 *>(this)->processRangeUntil(root, range, functor);
      }
    }

    template <typename RangeT>
    void rangeQueryIndices(RangeT const & range, std::vector<long> & result) const
    {
      if (root)
      {
        RangeQueryIndicesFunctor functor(result);
        const_cast<KDTree3 *>(this)->processRangeUntil(root, range, functor);
      }
    }

    /**
     * Apply a functor to all elements in a range, until the functor returns true. See the base class documentation
     * (RangeQueryStructure3::processRangeUntil()) for more information.
     *
     * The RangeT class should support intersection queries with AxisAlignedBox3 and containment queries with Vector3 and
     * AxisAlignedBox3.
     */
    template <typename RangeT, typename FunctorT>
    bool processRangeUntil(RangeT const & range, FunctorT & functor)
    {
      return root ? processRangeUntil(root, range, functor) : false;
    }

    bool rayIntersects(Ray3 const & ray, Real max_time = -1) const
    {
      return rayIntersectionTime(ray, max_time) >= 0;
    }

    Real rayIntersectionTime(Ray3 const & ray, Real max_time = -1) const
    {
      if (root)
      {
        if (root->bounds.rayIntersects(ray, max_time))
          return rayIntersectionTime(root, ray, max_time);
      }

      return -1;
    }

    RayStructureIntersection3 rayStructureIntersection(Ray3 const & ray, Real max_time = -1) const
    {
      if (root)
      {
        if (root->bounds.rayIntersects(ray, max_time))
          return rayStructureIntersection(root, ray, max_time);
      }

      return RayStructureIntersection3(-1);
    }

  private:
    /** Comparator for sorting elements along an axis. */
    struct ObjectLess
    {
      int coord;
      KDTree3 const * tree;

      /** Constructor. Axis 0 = X, 1 = Y, 2 = Z. */
      ObjectLess(int coord_, KDTree3 const * tree_) : coord(coord_), tree(tree_) {}

      /** Less-than operator, along the specified axis. */
      bool operator()(ElementIndex a, ElementIndex b)
      {
        // Compare object centers
        return BoundedObjectTraits::getCenter(tree->elems[a])[coord]
             < BoundedObjectTraits::getCenter(tree->elems[b])[coord];
      }
    };

    // Allow the comparator unrestricted access to the kd-tree.
    friend struct ObjectLess;

    /** A stack of element filters. */
    typedef std::vector<Filter<T> *> FilterStack;

    void moveIndicesToLeafPool(Node * leaf, IndexPool * main_index_pool, IndexPool * leaf_index_pool)
    {
      if (leaf)
      {
        ElementIndex * leaf_indices_start = leaf_index_pool->alloc(leaf->num_elems);
        std::memcpy(leaf_indices_start, leaf->elems, leaf->num_elems * sizeof(ElementIndex));
        leaf->elems = leaf_indices_start;

#if defined(_MSC_VER) && defined(_DEBUG)
        main_index_pool->_free(leaf->num_elems);
#else
        main_index_pool->free(leaf->num_elems);
#endif
      }
    }

    /** Recursively construct the tree. */
    void createTree(Node * start, bool save_memory, IndexPool * main_index_pool, IndexPool * leaf_index_pool)
    {
      // Assume the start node is fully constructed at this stage.
      //
      // If we are in memory-saving mode, then we assume the node's indices are last in the main index pool (but not yet in the
      // leaf pool, since we don't yet know if this node will turn out to be a leaf). In this case, after the function finishes,
      // the node's indices will be deallocated from the main index pool (and possibly moved to the leaf pool).

      if (!start || start->depth >= max_depth || (long)start->num_elems <= max_elems_in_leaf)
      {
        if (save_memory)
          moveIndicesToLeafPool(start, main_index_pool, leaf_index_pool);

        return;
      }

      // Find a splitting plane
#define THEA_KDTREE3_SPLIT_LONGEST
#ifdef THEA_KDTREE3_SPLIT_LONGEST
      Vector3 ext = start->bounds.extent();  // split longest dimension
      int coord = (ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2));
#else
      int coord = (int)(start->depth % 3);
#endif

// #define THEA_KDTREE3_SPLIT_MIDDLE
#ifdef THEA_KDTREE3_SPLIT_MIDDLE
      Real split = start->bounds.center()[coord];
#else
      size_t mid = start->num_elems / 2;
      std::nth_element(start->elems, start->elems + mid, start->elems + start->num_elems, ObjectLess(coord, this));
      Real split = BoundedObjectTraits::getCenter(elems[start->elems[mid]])[coord];
#endif

      start->lo = node_pool.alloc(1);
      start->lo->init(start->depth + 1);
      num_nodes++;

      start->hi = node_pool.alloc(1);
      start->hi->init(start->depth + 1);
      num_nodes++;

      // THEA_CONSOLE << "num_nodes = " << num_nodes;

      // It is critical (for compaction) that the child indices be allocated from a single buffer, i.e. in a single operation
      ElementIndex * child_indices_begin = main_index_pool->alloc(2 * start->num_elems);
      start->lo->elems = child_indices_begin;
      start->hi->elems = child_indices_begin + start->num_elems;

      static Real const SPLIT_FUDGE_FACTOR = 0.01f;
      Real node_size = start->bounds.high()[coord] - start->bounds.low()[coord];
      Real split_fudge = SPLIT_FUDGE_FACTOR * node_size;  // to handle numerical error at the split plane
      Real split_fudged_up = split + split_fudge;
      Real split_fudged_down = split - split_fudge;

      AxisAlignedBox3 elem_bounds;
      float m;
      bool lo_first = true, hi_first = true;
      for (ElementIndex i = 0; i < start->num_elems; ++i)
      {
        ElementIndex index = start->elems[i];
        BoundedObjectTraits::getBounds(elems[index], elem_bounds);

        m = elem_bounds.low()[coord];
        if (m <= split_fudged_up)
        {
          start->lo->elems[start->lo->num_elems++] = index;

          if (lo_first)
          {
            start->lo->bounds = elem_bounds;
            lo_first = false;
          }
          else
            start->lo->bounds.merge(elem_bounds);
        }

        m = elem_bounds.high()[coord];
        if (m >= split_fudged_down)
        {
          start->hi->elems[start->hi->num_elems++] = index;

          if (hi_first)
          {
            start->hi->bounds = elem_bounds;
            hi_first = false;
          }
          else
            start->hi->bounds.merge(elem_bounds);
        }
      }

      // If the split did not help at all, give up and call this a leaf
      if (start->lo->num_elems == start->num_elems || start->hi->num_elems == start->num_elems)
      {
#if defined(_MSC_VER) && defined(_DEBUG)
        node_pool._free(2);
#else
        node_pool.free(2);
#endif
        start->lo = start->hi = NULL;
        num_nodes -= 2;

        // THEA_CONSOLE << "num_nodes = " << num_nodes;

#if defined(_MSC_VER) && defined(_DEBUG)
        main_index_pool->_free(2 * start->num_elems);
#else
        main_index_pool->free(2 * start->num_elems);
#endif

        if (save_memory)
          moveIndicesToLeafPool(start, main_index_pool, leaf_index_pool);

        return;
      }

      // Compact the main index pool by shifting the high child indices, which are the last valid entries currently in the pool
      ElementIndex * lo_end = start->lo->elems + start->lo->num_elems;
      std::memmove(lo_end, start->hi->elems, (size_t)(start->hi->num_elems * sizeof(ElementIndex)));
      start->hi->elems = lo_end;

#if defined(_MSC_VER) && defined(_DEBUG)
        main_index_pool->_free(2 * start->num_elems - (start->lo->num_elems + start->hi->num_elems));
#else
        main_index_pool->free(2 * start->num_elems - (start->lo->num_elems + start->hi->num_elems));
#endif

      // The child nodes should be entirely contained within the parent node, and not intersect the split plane
      Vector3 slo = start->bounds.low();
      Vector3 shi = start->bounds.high();
      if (shi[coord] > split) shi[coord] = split;
      start->lo->bounds.set(start->lo->bounds.low().max(slo), start->lo->bounds.high().min(shi));

      shi[coord] = start->bounds.high()[coord];  // restore value
      if (slo[coord] < split) slo[coord] = split;
      start->hi->bounds.set(start->hi->bounds.low().max(slo), start->hi->bounds.high().min(shi));

      // Expand the bounding boxes slightly to handle numerical error
      start->lo->bounds.scaleCentered(BOUNDS_EXPANSION_FACTOR);
      start->hi->bounds.scaleCentered(BOUNDS_EXPANSION_FACTOR);

      // Recurse on the high child first, since its indices are at the end of the main index pool and can be freed first if
      // necessary
      createTree(start->hi, save_memory, main_index_pool, leaf_index_pool);

      // Recurse on the low child next, if we are in memory-saving mode its indices are now the last valid entries in the main
      // index pool
      createTree(start->lo, save_memory, main_index_pool, leaf_index_pool);

      // If we are in memory-saving mode, deallocate the indices stored at this node, which are currently the last entries in
      // the main index pool
      if (save_memory)
      {

#if defined(_MSC_VER) && defined(_DEBUG)
        main_index_pool->_free(start->num_elems);
#else
        main_index_pool->free(start->num_elems);
#endif

        
        start->num_elems = 0;
        start->elems = NULL;
      }
    }

    /** Mark that the bounding box requires an update. */
    void invalidateBounds()
    {
      valid_bounds = false;
    }

    /** Recompute the bounding box if it has been invalidated. */
    void updateBounds() const
    {
      if (valid_bounds) return;

      if (root)
        bounds = root->bounds;
      else
        bounds = AxisAlignedBox3::zero();

      valid_bounds = true;
    }

    /** Check if an element passes all filters currently on the stack. */
    bool elementPassesFilters(T const & elem) const
    {
      if (filters.empty()) return true;  // early exit

      for (typename FilterStack::const_iterator fi = filters.begin(); fi != filters.end(); ++fi)
        if (!(*fi)->allows(elem))
          return false;

      return true;
    }

    /** Get a bounding box for a node, in world space. */
    AxisAlignedBox3 const & getBoundsWorldSpace(Node const & node) const
    {
      return node.bounds;  // this version of the kd-tree doesn't support transforms
    }

    /**
     * Recursively look for the closest pair of points between two elements. Only pairs separated by less than the current
     * minimum distance will be considered.
     */
    template <typename MetricT, typename QueryT>
    IndexPair closestPair(Node const * start, QueryT const & query, AxisAlignedBox3 const & query_bounds,
                          double & mon_approx_dist, PointPair3 * closest_points) const
    {
      if (!start->lo)  // leaf
        return closestPairLeaf<MetricT>(start, query, mon_approx_dist, closest_points);
      else  // not leaf
      {
        // Figure out which child is closer (optimize for point queries?)
        Node const * n[2] = { start->lo, start->hi };
        double d[2] = { MetricT::monotoneApproxDistance(getBoundsWorldSpace(*n[0]), query_bounds),
                        MetricT::monotoneApproxDistance(getBoundsWorldSpace(*n[1]), query_bounds) };

        if (d[1] > d[0])
        {
          std::swap(n[0], n[1]);
          std::swap(d[0], d[1]);
        }

        IndexPair best_pair(-1, -1), pair;
        for (int i = 0; i < 2; ++i)
        {
          if (mon_approx_dist < 0 || d[i] <= mon_approx_dist)
          {
            pair = closestPair<MetricT>(n[i], query, query_bounds, mon_approx_dist, closest_points);
            if (pair.first >= 0 && pair.second >= 0) best_pair = pair;
          }
        }

        return best_pair;
      }
    }

    /**
     * Search the elements in a leaf node for the one closest to another element, when the latter is a proximity query
     * structure.
     */
    template <typename MetricT, typename QueryT>
    IndexPair closestPairLeaf(
      Node const * leaf,
      QueryT const & query,
      double & mon_approx_dist,
      PointPair3 * closest_points,
      typename boost::enable_if<boost::is_base_of<ProximityQueryBaseT, QueryT>, void>::type * dummy = NULL) const
    {
      PointPair3 cpp;
      PointPair3 * cpp_ptr = closest_points ? &cpp : NULL;
      IndexPair best_pair(-1, -1), pair;
      double d = -1;

      for (size_t i = 0; i < leaf->num_elems; ++i)
      {
        ElementIndex index = leaf->elems[i];
        Element const & elem = elems[index];

        if (!elementPassesFilters(elem))
          continue;

        pair = query.closestPair<MetricT>(elem, mon_approx_dist, &d, cpp_ptr);

        if (pair.first >= 0 && pair.second >= 0)
        {
          best_pair = IndexPair((long)index, pair.first);

          mon_approx_dist = d;
          if (closest_points) *closest_points = cpp;
        }
      }

      return best_pair;
    }

    /**
     * Search the elements in a leaf node for the one closest to another element, when the latter is NOT a proximity query
     * structure.
     */
    template <typename MetricT, typename QueryT>
    IndexPair closestPairLeaf(
      Node const * leaf,
      QueryT const & query,
      double & mon_approx_dist,
      PointPair3 * closest_points,
      typename boost::disable_if<boost::is_base_of<ProximityQueryBaseT, QueryT>, void>::type * dummy = NULL) const
    {
      PointPair3 cpp;
      IndexPair best_pair(-1, -1);
      double d;

      for (size_t i = 0; i < leaf->num_elems; ++i)
      {
        ElementIndex index = leaf->elems[i];
        Element const & elem = elems[index];

        if (!elementPassesFilters(elem))
          continue;

        d = MetricT::closestPoints(elem, query, cpp.first, cpp.second);

        if (mon_approx_dist < 0 || d <= mon_approx_dist)
        {
          best_pair = IndexPair((long)index, 0);

          mon_approx_dist = d;
          if (closest_points) *closest_points = cpp;
        }
      }

      return best_pair;
    }

    /**
     * Recursively look for the k closest elements to a query object. Only elements at less than the specified maximum distance
     * will be considered.
     */
    template <typename MetricT, typename QueryT>
    void kClosestElements(Node const * start, QueryT const & query, AxisAlignedBox3 const & query_bounds,
                          BoundedSortedArray<Neighbor> & k_closest_elems, double dist_bound) const
    {
      if (!start->lo)  // leaf
        kClosestElementsLeaf<MetricT>(start, query, k_closest_elems, dist_bound);
      else  // not leaf
      {
        // Figure out which child is closer (optimize for point queries?)
        Node const * n[2] = { start->lo, start->hi };
        double d[2] = { MetricT::monotoneApproxDistance(getBoundsWorldSpace(*n[0]), query_bounds),
                        MetricT::monotoneApproxDistance(getBoundsWorldSpace(*n[1]), query_bounds) };

        if (d[1] > d[0])
        {
          std::swap(n[0], n[1]);
          std::swap(d[0], d[1]);
        }

        double mon_approx_dist_bound = (dist_bound >= 0 ? MetricT::computeMonotoneApprox(dist_bound) : -1);

        for (int i = 0; i < 2; ++i)
          if ((mon_approx_dist_bound < 0 || d[i] <= mon_approx_dist_bound) && k_closest_elems.isInsertable(Neighbor(d[i], 0)))
            kClosestElements<MetricT>(n[i], query, query_bounds, k_closest_elems, dist_bound);
      }
    }

    /**
     * Search the elements in a leaf node for the k nearest neighbors of an object, when the latter is a proximity query
     * structure.
     */
    template <typename MetricT, typename QueryT>
    void kClosestElementsLeaf(
      Node const * leaf,
      QueryT const & query,
      BoundedSortedArray<Neighbor> & k_closest_elems,
      double dist_bound,
      typename boost::enable_if<boost::is_base_of<ProximityQueryBaseT, QueryT>, void>::type * dummy = NULL) const
    {
      for (size_t i = 0; i < leaf->num_elems; ++i)
      {
        ElementIndex index = leaf->elems[i];
        Element const & elem = elems[index];

        if (elementPassesFilters(elem))
          query.kClosestElements<MetricT>(elem, k_closest_elems, dist_bound, false);
      }
    }

    /**
     * Search the elements in a leaf node for the one closest to another element, when the latter is NOT a proximity query
     * structure.
     */
    template <typename MetricT, typename QueryT>
    void kClosestElementsLeaf(
      Node const * leaf,
      QueryT const & query,
      BoundedSortedArray<Neighbor> & k_closest_elems,
      double dist_bound,
      typename boost::disable_if<boost::is_base_of<ProximityQueryBaseT, QueryT>, void>::type * dummy = NULL) const
    {
      double mon_approx_dist_bound = (dist_bound >= 0 ? MetricT::computeMonotoneApprox(dist_bound) : -1);

      PointPair3 cpp;
      double d;
      bool found;

      for (size_t i = 0; i < leaf->num_elems; ++i)
      {
        ElementIndex index = leaf->elems[i];
        Element const & elem = elems[index];

        if (!elementPassesFilters(elem))
          continue;

        // We need to do an explicit test because the less-than comparator on floats is not sufficient for judging equality
        // (and just imposing equality by comparing indices of neighbors breaks the complete ordering). We'll assume this test
        // is faster, for general objects, than the actual distance computation which happens next.
        found = false;
        for (int j = 0; j < k_closest_elems.size(); ++j)
          if (k_closest_elems[j].getIndex() == (long)index)
          {
            found = true;
            break;
          }

        if (found) continue;

        d = MetricT::closestPoints(elem, query, cpp.first, cpp.second);

        if (mon_approx_dist_bound < 0 || d <= mon_approx_dist_bound)
          k_closest_elems.insert(Neighbor(d, (long)index));
      }
    }

    /**
     * Apply a functor to all elements of a subtree within a range, stopping when the functor returns true on any point. The
     * RangeT class should support containment queries with AxisAlignedBox3.
     *
     * @return True if the functor evaluated to true on any point in the range, else false.
     */
    template <typename RangeT, typename FunctorT>
    bool processRangeUntil(Node const * start, RangeT const & range, FunctorT & functor)
    {
      // Early exit if the range and node are disjoint
      AxisAlignedBox3 tr_start_bounds = getBoundsWorldSpace(*start);
      if (!IntersectionTester::intersects(range, tr_start_bounds))
        return false;

      // If the entire node is contained in the range AND there are element references at this node (so it's either a leaf or we
      // have not saved memory by flushing references at internal nodes), then process all these elems.
      if (start->num_elems > 0 && range.contains(tr_start_bounds))
      {
        for (size_t i = 0; i < start->num_elems; ++i)
        {
          ElementIndex index = start->elems[i];
          Element & elem = elems[index];

          if (!elementPassesFilters(elem))
            continue;

          if (functor(static_cast<long>(index), elem))
            return true;
        }
      }
      else if (!start->lo)  // leaf
      {
        for (size_t i = 0; i < start->num_elems; ++i)
        {
          ElementIndex index = start->elems[i];
          Element & elem = elems[index];

          if (!elementPassesFilters(elem))
            continue;

          bool intersects = IntersectionTester::intersects(elem, range);
          if (intersects)
            if (functor(static_cast<long>(index), elem))
              return true;
        }
      }
      else  // not leaf
      {
        if (processRangeUntil(start->lo, range, functor)) return true;
        if (processRangeUntil(start->hi, range, functor)) return true;
      }

      return false;
    }

    /** Transform a ray to local/object space. */
    Ray3 const & toObjectSpace(Ray3 const & ray) const
    {
      return ray;  // this version of the kd-tree doesn't support transforms
    }

    /** Transform a normal to world space. */
    Vector3 const & normalToWorldSpace(Vector3 const & n) const
    {
      return n;  // this version of the kd-tree doesn't support transforms
    }

    /**
     * Check if the ray intersection time \a new_time represents a closer, or equally close, valid hit than the previous best
     * time \a old_time.
     */
    static bool improvedRayTime(Real new_time, Real old_time)
    {
      return (new_time >= 0 && (old_time < 0 || new_time <= old_time));
    }

    /** Get the time taken for a ray to hit the nearest object in a node, in the forward direction. */
    Real rayIntersectionTime(Node const * start, Ray3 const & ray, Real max_time) const
    {
      if (!start->lo)  // leaf
      {
        Real best_time = max_time;
        bool found = false;
        for (size_t i = 0; i < start->num_elems; ++i)
        {
          ElementIndex index = start->elems[i];
          Element const & elem = elems[index];

          if (!elementPassesFilters(elem))
            continue;

          Real time = RayIntersectionTester::rayIntersectionTime(ray, elem, best_time);
          if (improvedRayTime(time, best_time))
          {
            best_time = time;
            found = true;
          }
        }

        return found ? best_time : -1;
      }
      else  // not leaf
      {
        // Figure out which child will be hit first
        Node const * n[2] = { start->lo, start->hi };
        Real t[2] = { n[0]->bounds.rayIntersectionTime(ray, max_time),
                      n[1]->bounds.rayIntersectionTime(ray, max_time) };

        if (t[0] < 0 && t[1] < 0)
          return -1;

        if (improvedRayTime(t[1], t[0]))
        {
          std::swap(n[0], n[1]);
          std::swap(t[0], t[1]);
        }

        Real best_time = max_time;
        bool found = false;
        for (int i = 0; i < 2; ++i)
        {
          if (improvedRayTime(t[i], best_time))
          {
            Real time = rayIntersectionTime(n[i], ray, best_time);
            if (improvedRayTime(time, best_time))
            {
              best_time = time;
              found = true;
            }
          }
        }

        return found ? best_time : -1;
      }
    }

    /** Get the nearest intersection of a ray with a node in the forward direction. */
    RayStructureIntersection3 rayStructureIntersection(Node const * start, Ray3 const & ray, Real max_time) const
    {
      if (!start->lo)  // leaf
      {
        RayStructureIntersection3 best_isec(max_time);
        bool found = false;
        for (size_t i = 0; i < start->num_elems; ++i)
        {
          ElementIndex index = start->elems[i];
          Element const & elem = elems[index];

          if (!elementPassesFilters(elem))
            continue;

          RayIntersection3 isec = RayIntersectionTester::rayIntersection(ray, elem, best_isec.getTime());
          if (improvedRayTime(isec.getTime(), best_isec.getTime()))
          {
            best_isec = RayStructureIntersection3(isec, (long)index);
            found = true;
          }
        }

        return found ? best_isec : RayStructureIntersection3(-1);
      }
      else  // not leaf
      {
        // Figure out which child will be hit first
        Node const * n[2] = { start->lo, start->hi };
        Real t[2] = { n[0]->bounds.rayIntersectionTime(ray, max_time),
                      n[1]->bounds.rayIntersectionTime(ray, max_time) };

        if (t[0] < 0 && t[1] < 0)
          return -1;

        if (improvedRayTime(t[1], t[0]))
        {
          std::swap(n[0], n[1]);
          std::swap(t[0], t[1]);
        }

        RayStructureIntersection3 best_isec(max_time);
        bool found = false;
        for (int i = 0; i < 2; ++i)
        {
          if (improvedRayTime(t[i], best_isec.getTime()))
          {
            RayStructureIntersection3 isec = rayStructureIntersection(n[i], ray, best_isec.getTime());
            if (improvedRayTime(isec.getTime(), best_isec.getTime()))
            {
              best_isec = isec;
              found = true;
            }
          }
        }

        return found ? best_isec : RayStructureIntersection3(-1);
      }
    }

    Node * root;

    long num_elems;  // elems.size() doesn't tell us how many elements there are, it's just the capacity of the elems array
    ElementArray elems;  // elems.size() is *not* the number of elements in the tree!!!

    long num_nodes;
    NodePool node_pool;

    IndexPool index_pool;

    int max_depth;
    int max_elems_in_leaf;

    FilterStack filters;

    mutable bool valid_bounds;
    mutable AxisAlignedBox3 bounds;

    static Real const BOUNDS_EXPANSION_FACTOR;

}; // class KDTree3

// Static variables
template <typename T> Real const KDTree3<T>::BOUNDS_EXPANSION_FACTOR = 1.05f;

} // namespace Thea

#endif
